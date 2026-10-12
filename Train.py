import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random
import itertools
import sys

from bigram import (
    GPTLanguageModel, DASAugmentedGPT, VoxelLattice, WormholeRouter,
    ConformalRouter, morton_encode_4d, morton_decode_4d,
    device, block_size, n_embd, n_head, n_layer, dropout
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

use_cuda = device == 'cuda'

if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if use_cuda:
    batch_size = 64
    max_iters = 3000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 20
    lattice_dims = (5, 5, 5, 5)
    seq_len = 64
    num_seqs = 8000
else:
    batch_size = 16
    max_iters = 500
    eval_interval = 500
    learning_rate = 1e-3
    eval_iters = 5
    lattice_dims = (3, 3, 3, 3)
    seq_len = 32
    num_seqs = 2000

warmup_steps = max(1, max_iters // 20)
amp_enabled = use_cuda
amp_dtype = torch.bfloat16 if use_cuda else torch.float32

try:
    if use_cuda:
        _test_p = torch.nn.Parameter(torch.zeros(1, device=device))
        _test_opt = torch.optim.AdamW([_test_p], fused=True)
        del _test_opt, _test_p
        fused_available = True
    else:
        fused_available = False
except Exception:
    fused_available = False

use_compile = (
    sys.platform.startswith('linux') and
    use_cuda and
    hasattr(torch, 'compile')
)


def generate_4d_lattice_dataset(dims, seq_length, n_sequences, held_out_fraction=0.15, seed=42):
    rng = random.Random(seed)
    all_coords = list(itertools.product(*(range(d) for d in dims)))
    vocab_tokens = []
    coord_to_token = {}
    token_to_coord = {}
    for c in all_coords:
        label = f"{c[0]}{c[1]}{c[2]}{c[3]}"
        if label not in vocab_tokens:
            vocab_tokens.append(label)
        coord_to_token[c] = vocab_tokens.index(label)
        token_to_coord[vocab_tokens.index(label)] = c
    noise_token = len(vocab_tokens)
    vocab_tokens.append("N")
    vocab_size = len(vocab_tokens)

    coord_pairs = []
    for c in all_coords:
        for axis in range(4):
            for delta in (-1, 1):
                nb = list(c)
                nb[axis] += delta
                if 0 <= nb[axis] < dims[axis]:
                    coord_pairs.append((c, tuple(nb)))

    n_held = max(1, int(len(coord_pairs) * held_out_fraction))
    rng.shuffle(coord_pairs)
    held_out_set = set()
    for p in coord_pairs[:n_held]:
        held_out_set.add(p)
    train_pairs = [p for p in coord_pairs if p not in held_out_set]
    val_pairs = list(held_out_set)

    def make_sequences(pairs, n_seq, noise_rate=0.05):
        pair_set = set(pairs)
        seqs = []
        coord_seqs = []
        for _ in range(n_seq):
            seq = []
            cseq = []
            start_pair = rng.choice(pairs)
            current = start_pair[0]
            seq.append(coord_to_token[current])
            cseq.append(list(current))
            for _ in range(seq_length - 1):
                if rng.random() < noise_rate:
                    seq.append(noise_token)
                    cseq.append(list(current))
                else:
                    neighbors = []
                    for axis in range(4):
                        for delta in (-1, 1):
                            nb = list(current)
                            nb[axis] += delta
                            if 0 <= nb[axis] < dims[axis]:
                                nb_tuple = tuple(nb)
                                if (current, nb_tuple) in pair_set:
                                    neighbors.append(nb_tuple)
                    if not neighbors:
                        neighbors = [current]
                    nxt = rng.choice(neighbors)
                    seq.append(coord_to_token[nxt])
                    cseq.append(list(nxt))
                    current = nxt
            seqs.append(seq)
            coord_seqs.append(cseq)
        return seqs, coord_seqs

    n_train = int(n_sequences * 0.85)
    n_val = n_sequences - n_train
    train_seqs, train_coords = make_sequences(train_pairs, n_train)
    val_seqs, val_coords = make_sequences(val_pairs, n_val)

    train_x = torch.tensor([s[:-1] for s in train_seqs], dtype=torch.long)
    train_y = torch.tensor([s[1:] for s in train_seqs], dtype=torch.long)
    train_c = torch.tensor([c[:-1] for c in train_coords], dtype=torch.long)
    val_x = torch.tensor([s[:-1] for s in val_seqs], dtype=torch.long)
    val_y = torch.tensor([s[1:] for s in val_seqs], dtype=torch.long)
    val_c = torch.tensor([c[:-1] for c in val_coords], dtype=torch.long)

    return train_x, train_y, train_c, val_x, val_y, val_c, vocab_size, vocab_tokens, token_to_coord


train_x, train_y, train_c, val_x, val_y, val_c, vocab_size, vocab_tokens, token_to_coord = \
    generate_4d_lattice_dataset(lattice_dims, seq_len, num_seqs)

lattice = VoxelLattice(lattice_dims)
wrouter = WormholeRouter(lattice, seed=42)

train_wf = torch.zeros(train_c.shape[0], train_c.shape[1], dtype=torch.long)
for i in range(train_c.shape[0]):
    for t in range(train_c.shape[1]):
        coords = tuple(train_c[i, t].tolist())
        code = lattice.get_code(coords)
        train_wf[i, t] = 1 if wrouter.has_wormhole(code) else 0

val_wf = torch.zeros(val_c.shape[0], val_c.shape[1], dtype=torch.long)
for i in range(val_c.shape[0]):
    for t in range(val_c.shape[1]):
        coords = tuple(val_c[i, t].tolist())
        code = lattice.get_code(coords)
        val_wf[i, t] = 1 if wrouter.has_wormhole(code) else 0

train_x = train_x.to(device)
train_y = train_y.to(device)
train_c = train_c.to(device)
train_wf = train_wf.to(device)
val_x = val_x.to(device)
val_y = val_y.to(device)
val_c = val_c.to(device)
val_wf = val_wf.to(device)


def get_batch(split, bsz=None):
    bsz = bsz or batch_size
    if split == 'train':
        dx, dy, dc, dw = train_x, train_y, train_c, train_wf
    else:
        dx, dy, dc, dw = val_x, val_y, val_c, val_wf
    ix = torch.randint(len(dx), (bsz,), device=device)
    return dx[ix], dy[ix], dc[ix], dw[ix]


ARM_VANILLA = 0
ARM_LATTICE = 1
ARM_WORMHOLE = 2
ARM_FULL = 3


@torch.no_grad()
def estimate_loss(model, n_batches=None, arm_type=ARM_VANILLA):
    n_batches = n_batches or eval_iters
    model.eval()
    out = {}
    for split in ['train', 'val']:
        total_loss = 0.0
        for _ in range(n_batches):
            xb, yb, cb, wb = get_batch(split)
            with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                                dtype=amp_dtype, enabled=amp_enabled):
                if arm_type == ARM_VANILLA:
                    _, loss = model(xb, yb)
                elif arm_type == ARM_LATTICE:
                    _, loss = model(xb, yb, coord_indices=cb)
                elif arm_type == ARM_WORMHOLE:
                    _, loss = model(xb, yb, wormhole_flags=wb)
                elif arm_type == ARM_FULL:
                    _, loss = model(xb, yb, coord_indices=cb, wormhole_flags=wb)
            total_loss += loss.detach()
        out[split] = (total_loss / n_batches).item()
    model.train()
    return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_lr(step, max_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(model, arm_name, seed, iters=None, arm_type=ARM_VANILLA):
    iters = iters or max_iters
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    opt_kwargs = dict(lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    if fused_available:
        opt_kwargs['fused'] = True
    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    compiled_model = model
    if use_compile:
        try:
            compiled_model = torch.compile(model)
        except Exception:
            compiled_model = model

    losses = None
    for step in range(iters):
        xb, yb, cb, wb = get_batch('train')
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            if arm_type == ARM_VANILLA:
                _, loss = compiled_model(xb, yb)
            elif arm_type == ARM_LATTICE:
                _, loss = compiled_model(xb, yb, coord_indices=cb)
            elif arm_type == ARM_WORMHOLE:
                _, loss = compiled_model(xb, yb, wormhole_flags=wb)
            elif arm_type == ARM_FULL:
                _, loss = compiled_model(xb, yb, coord_indices=cb, wormhole_flags=wb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        lr = get_lr(step, iters, learning_rate, warmup_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if (step + 1) % eval_interval == 0:
            losses = estimate_loss(model, arm_type=arm_type)
            print(f"{arm_name}, seed {seed}, step {step + 1}: "
                  f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if losses is None or iters % eval_interval != 0:
        losses = estimate_loss(model, arm_type=arm_type)
    return losses['val'], losses['train']


if use_cuda:
    exp_embd = n_embd
    exp_heads = n_head
    exp_layers = n_layer
    exp_drop = dropout
else:
    exp_embd = 64
    exp_heads = 4
    exp_layers = 3
    exp_drop = 0.0

exp_block = seq_len - 1

seeds = [42, 137, 2024]

dummy_vanilla = GPTLanguageModel(vocab_size, embd_dim=exp_embd, num_heads=exp_heads,
                                  num_layers=exp_layers, blk_size=exp_block, drop=exp_drop).to(device)
vanilla_params = count_params(dummy_vanilla)
del dummy_vanilla

dummy_full = DASAugmentedGPT(
    vocab_size, lattice_dims,
    embd_dim=exp_embd, num_heads=exp_heads, num_layers=exp_layers,
    blk_size=exp_block, drop=exp_drop,
    use_lattice_emb=True, use_wormhole_feat=True
).to(device)
hybrid_params = count_params(dummy_full)
del dummy_full

matched_layers = exp_layers
matched_embd = exp_embd
while True:
    dm = GPTLanguageModel(vocab_size, embd_dim=matched_embd, num_heads=exp_heads,
                          num_layers=matched_layers, blk_size=exp_block, drop=exp_drop).to(device)
    mp = count_params(dm)
    del dm
    if mp >= hybrid_params * 0.95:
        break
    matched_embd += exp_heads
    if matched_embd > exp_embd * 4:
        matched_layers += 1
        matched_embd = exp_embd
    if matched_layers > exp_layers * 4:
        break

if use_cuda:
    torch.cuda.empty_cache()

print(f"Device: {device}, Vanilla parameters: {vanilla_params:,}")

all_codes = list(lattice.nodes.keys())

passed = True
for code, coords in list(lattice.nodes.items())[:50]:
    re_encoded = morton_encode_4d(*coords)
    decoded = morton_decode_4d(code)
    if re_encoded != code or decoded != coords:
        passed = False
        break
print(f"Morton encode/decode round-trip: {'PASS' if passed else 'FAIL'}")

rng_val = random.Random(99)
sampled = rng_val.sample(all_codes, min(40, len(all_codes)))
plain_dists = []
worm_dists = []
for i in range(len(sampled)):
    for j in range(i + 1, len(sampled)):
        pd = lattice.manhattan_distance(sampled[i], sampled[j])
        wd = wrouter.shortest_path_length(sampled[i], sampled[j])
        plain_dists.append(pd)
        worm_dists.append(wd)
avg_plain = sum(plain_dists) / len(plain_dists)
avg_worm = sum(worm_dists) / len(worm_dists)
print(f"Wormhole routing avg path length - plain: {avg_plain:.2f}, wormholes: {avg_worm:.2f}")

alpha = 0.1
conf = ConformalRouter(alpha=alpha)
rng_conf = random.Random(123)
cal_scores = [rng_conf.gauss(0, 1) for _ in range(500)]
test_scores = [rng_conf.gauss(0, 1) for _ in range(200)]
conf.calibrate(cal_scores)
empirical_cov = conf.coverage(test_scores)
target_cov = 1.0 - alpha
print(f"Conformal routing coverage - target: {target_cov:.2f}, empirical: {empirical_cov:.4f}")

total_nodes = len(lattice.nodes)
total_edges = sum(len(adj) for adj in lattice.adjacency.values())
max_edges = total_nodes * (total_nodes - 1)
sparsity = 1.0 - (total_edges / max_edges) if max_edges > 0 else 1.0
print(f"Lattice sparsity: {sparsity:.6f} ({total_nodes} nodes, {total_edges} edges)")


def make_vanilla():
    return GPTLanguageModel(vocab_size, embd_dim=exp_embd, num_heads=exp_heads,
                            num_layers=exp_layers, blk_size=exp_block, drop=exp_drop).to(device)


def make_matched():
    return GPTLanguageModel(vocab_size, embd_dim=matched_embd, num_heads=exp_heads,
                            num_layers=matched_layers, blk_size=exp_block, drop=exp_drop).to(device)


def make_lattice_only():
    return DASAugmentedGPT(
        vocab_size, lattice_dims,
        embd_dim=exp_embd, num_heads=exp_heads, num_layers=exp_layers,
        blk_size=exp_block, drop=exp_drop,
        use_lattice_emb=True, use_wormhole_feat=False
    ).to(device)


def make_wormhole_only():
    return DASAugmentedGPT(
        vocab_size, lattice_dims,
        embd_dim=exp_embd, num_heads=exp_heads, num_layers=exp_layers,
        blk_size=exp_block, drop=exp_drop,
        use_lattice_emb=False, use_wormhole_feat=True
    ).to(device)


def make_full_hybrid():
    return DASAugmentedGPT(
        vocab_size, lattice_dims,
        embd_dim=exp_embd, num_heads=exp_heads, num_layers=exp_layers,
        blk_size=exp_block, drop=exp_drop,
        use_lattice_emb=True, use_wormhole_feat=True
    ).to(device)


arm_specs = [
    ("Vanilla", make_vanilla, ARM_VANILLA),
    ("Vanilla-Matched", make_matched, ARM_VANILLA),
    ("Vanilla+Lattice", make_lattice_only, ARM_LATTICE),
    ("Vanilla+Wormhole", make_wormhole_only, ARM_WORMHOLE),
    ("Full-Hybrid", make_full_hybrid, ARM_FULL),
]

saved_vanilla = None
results = {}
for arm_name, make_fn, arm_type in arm_specs:
    results[arm_name] = {'val_losses': [], 'train_losses': [], 'params': 0}
    for seed in seeds:
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
        model = make_fn()
        results[arm_name]['params'] = count_params(model)
        val_loss, train_loss = train_model(model, arm_name, seed, arm_type=arm_type)
        results[arm_name]['val_losses'].append(val_loss)
        results[arm_name]['train_losses'].append(train_loss)
        if arm_name == "Vanilla" and saved_vanilla is None:
            saved_vanilla = model
        else:
            del model
        if use_cuda:
            torch.cuda.empty_cache()

header = f"{'Arm':<22} {'Params':>10} {'Val Loss':>16} {'BPC':>16} {'Train Loss':>16}"
print(header)
print("-" * 90)

for arm_name in [a[0] for a in arm_specs]:
    r = results[arm_name]
    vl = r['val_losses']
    tl = r['train_losses']
    mean_vl = sum(vl) / len(vl)
    std_vl = (sum((x - mean_vl) ** 2 for x in vl) / len(vl)) ** 0.5
    mean_tl = sum(tl) / len(tl)
    std_tl = (sum((x - mean_tl) ** 2 for x in tl) / len(tl)) ** 0.5
    bpc_vals = [v / math.log(2) for v in vl]
    mean_bpc = sum(bpc_vals) / len(bpc_vals)
    std_bpc = (sum((x - mean_bpc) ** 2 for x in bpc_vals) / len(bpc_vals)) ** 0.5
    params = r['params']
    print(f"{arm_name:<22} {params:>10,} "
          f"{mean_vl:>7.4f} +/- {std_vl:.4f} "
          f"{mean_bpc:>7.4f} +/- {std_bpc:.4f} "
          f"{mean_tl:>7.4f} +/- {std_tl:.4f}")

axes = ['X', 'Y', 'Z', 'W']
torch.manual_seed(42)
saved_vanilla.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    lattice_tokens = saved_vanilla.generate(context, max_new_tokens=15)[0].tolist()

last_coord = None
for step, tok in enumerate(lattice_tokens):
    if tok in token_to_coord:
        c = token_to_coord[tok]
        if last_coord is None:
            print(f"Step {step}:  {c}")
        else:
            diffs = []
            for a_idx, (curr, prev) in enumerate(zip(c, last_coord)):
                d = curr - prev
                if d != 0:
                    sign = '+' if d > 0 else ''
                    diffs.append(f"{sign}{d} along {axes[a_idx]}-axis")
            movement = ', '.join(diffs) if diffs else 'same coordinate'
            print(f"Step {step}:  {c}   [{movement}]")
        last_coord = c
    else:
        print(f"Step {step}:  [NOISE TOKEN]   (perturbed state)")
del saved_vanilla

print()
with open('john_wick.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

chars = sorted(list(set(raw_text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
text_data = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long, device=device)

gen_model = GPTLanguageModel(len(chars), embd_dim=exp_embd, num_heads=exp_heads,
                             num_layers=exp_layers, blk_size=exp_block, drop=exp_drop).to(device)
gen_opt = torch.optim.AdamW(gen_model.parameters(), lr=1e-3)

for _ in range(300):
    ix = torch.randint(len(text_data) - exp_block - 1, (batch_size,), device=device)
    xb = torch.stack([text_data[i:i + exp_block] for i in ix])
    yb = torch.stack([text_data[i + 1:i + exp_block + 1] for i in ix])
    _, loss = gen_model(xb, yb)
    gen_opt.zero_grad(set_to_none=True)
    loss.backward()
    gen_opt.step()

gen_model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    tokens_out = gen_model.generate(context, max_new_tokens=150)[0].tolist()
print(''.join(itos.get(t, '') for t in tokens_out).strip())
del gen_model


