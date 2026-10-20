import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random
import itertools
import sys
import time

from bigram import (
    GPTLanguageModel, DASAugmentedGPT, VoxelLattice, WormholeRouter,
    ConformalRouter, morton_encode_4d, morton_decode_4d,
    device, block_size, n_embd, n_head, n_layer, dropout
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

start_time = time.time()

use_cuda = device == 'cuda'

if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    torch.set_num_threads(1)

if use_cuda:
    batch_size = 64
    max_iters = 3000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 20
    lattice_dims = (5, 5, 5, 5)
    seq_len = 64
    num_seqs = 6000
    exp_embd = n_embd
    exp_heads = n_head
    exp_layers = n_layer
    exp_drop = dropout
else:
    batch_size = 16
    max_iters = 2000
    eval_interval = 500
    learning_rate = 1e-3
    eval_iters = 5
    lattice_dims = (3, 3, 3, 3)
    seq_len = 24
    num_seqs = 1200
    exp_embd = 48
    exp_heads = 4
    exp_layers = 2
    exp_drop = 0.0

seeds = [42, 137, 2024, 7, 99] if use_cuda else [42, 137, 2024]
if not use_cuda:
    print("CPU run: seeds reduced from 5 to 3 to keep the total runtime under 20 minutes")

exp_block = seq_len - 1
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_lr(step, max_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = (sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5
    return m, s


def generate_seed_data(dims, seq_length, n_sequences, seed, held_out_fraction=0.15):
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
    token_to_coord[noise_token] = tuple(dims)

    lattice_adj = {c: [] for c in all_coords}
    coord_pairs = []
    for c in all_coords:
        for axis in range(4):
            for delta in (-1, 1):
                nb = list(c)
                nb[axis] += delta
                if 0 <= nb[axis] < dims[axis]:
                    nb_t = tuple(nb)
                    lattice_adj[c].append(nb_t)
                    coord_pairs.append((c, nb_t))

    undirected_edges = []
    seen_edges = set()
    for u, v in coord_pairs:
        key = (min(u, v), max(u, v))
        if key not in seen_edges:
            seen_edges.add(key)
            undirected_edges.append(key)
    n_held = max(1, int(len(undirected_edges) * held_out_fraction))
    rng.shuffle(undirected_edges)
    held_out_set = set()
    for u, v in undirected_edges[:n_held]:
        held_out_set.add((u, v))
        held_out_set.add((v, u))
    train_pairs = [p for p in coord_pairs if p not in held_out_set]
    train_adj = {c: [] for c in all_coords}
    for u, v in train_pairs:
        train_adj[u].append(v)
    valid_train_starts = [c for c in all_coords if len(train_adj[c]) > 0]

    noise_rate = 0.05
    n_train = int(n_sequences * 0.85)
    n_val = n_sequences - n_train

    train_seqs = []
    for _ in range(n_train):
        seq = []
        current = rng.choice(valid_train_starts)
        seq.append(coord_to_token[current])
        for _ in range(seq_length - 1):
            if rng.random() < noise_rate:
                seq.append(noise_token)
            else:
                nxt = rng.choice(train_adj[current])
                seq.append(coord_to_token[nxt])
                current = nxt
        train_seqs.append(seq)

    for _s in train_seqs:
        for _t in range(len(_s) - 1):
            if _s[_t] != noise_token and _s[_t + 1] != noise_token:
                assert (token_to_coord[_s[_t]], token_to_coord[_s[_t + 1]]) not in held_out_set

    val_seqs = []
    for _ in range(n_val):
        seq = []
        current = rng.choice(all_coords)
        seq.append(coord_to_token[current])
        for _ in range(seq_length - 1):
            if rng.random() < noise_rate:
                seq.append(noise_token)
            else:
                nxt = rng.choice(lattice_adj[current])
                seq.append(coord_to_token[nxt])
                current = nxt
        val_seqs.append(seq)

    train_x = torch.tensor([s[:-1] for s in train_seqs], dtype=torch.long)
    train_y = torch.tensor([s[1:] for s in train_seqs], dtype=torch.long)
    val_x = torch.tensor([s[:-1] for s in val_seqs], dtype=torch.long)
    val_y = torch.tensor([s[1:] for s in val_seqs], dtype=torch.long)

    val_mask = torch.zeros(val_x.shape, dtype=torch.bool)
    val_train_mask = torch.zeros(val_x.shape, dtype=torch.bool)
    for i in range(len(val_seqs)):
        for t in range(seq_length - 1):
            tok_u = val_seqs[i][t]
            tok_v = val_seqs[i][t + 1]
            if tok_u != noise_token and tok_v != noise_token:
                c_u = token_to_coord[tok_u]
                c_v = token_to_coord[tok_v]
                if (c_u, c_v) in held_out_set:
                    val_mask[i, t] = True
                else:
                    val_train_mask[i, t] = True

    lattice = VoxelLattice(dims)
    wrouter = WormholeRouter(lattice, seed=seed)

    coord_table = torch.tensor([token_to_coord[i] for i in range(vocab_size)], dtype=torch.long)
    wf_table = torch.tensor([
        1 if t < noise_token and wrouter.has_wormhole(lattice.get_code(token_to_coord[t])) else 0
        for t in range(vocab_size)
    ], dtype=torch.long)

    rng_perm = random.Random(12345)
    node_perm = list(range(len(all_coords)))
    rng_perm.shuffle(node_perm)

    ctrl_coord_table = coord_table.clone()
    ctrl_wf_table = wf_table.clone()
    for i in range(len(all_coords)):
        ctrl_coord_table[i] = coord_table[node_perm[i]]
        ctrl_wf_table[i] = wf_table[node_perm[i]]

    train_c = coord_table[train_x]
    val_c = coord_table[val_x]
    train_wf = wf_table[train_x]
    val_wf = wf_table[val_x]

    train_c_shuf = ctrl_coord_table[train_x]
    val_c_shuf = ctrl_coord_table[val_x]
    train_wf_rand = ctrl_wf_table[train_x]
    val_wf_rand = ctrl_wf_table[val_x]

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    train_c = train_c.to(device)
    train_c_shuf = train_c_shuf.to(device)
    train_wf = train_wf.to(device)
    train_wf_rand = train_wf_rand.to(device)

    val_x = val_x.to(device)
    val_y = val_y.to(device)
    val_c = val_c.to(device)
    val_c_shuf = val_c_shuf.to(device)
    val_wf = val_wf.to(device)
    val_wf_rand = val_wf_rand.to(device)
    val_mask = val_mask.to(device)
    val_train_mask = val_train_mask.to(device)
    coord_table = coord_table.to(device)
    ctrl_coord_table = ctrl_coord_table.to(device)
    wf_table = wf_table.to(device)
    ctrl_wf_table = ctrl_wf_table.to(device)

    uniform_baseline = math.log(vocab_size)
    scored_coords = [token_to_coord[val_seqs[i][t]] for i in range(len(val_seqs)) for t in range(seq_length - 1) if val_mask[i, t].item()]
    neighbor_baseline = sum(math.log(len(lattice_adj[c])) for c in scored_coords) / max(1, len(scored_coords))

    train_data = (train_x, train_y, train_c, train_c_shuf, train_wf, train_wf_rand)
    val_data = (val_x, val_y, val_c, val_c_shuf, val_wf, val_wf_rand)

    return (train_data, val_data, val_mask, val_train_mask, vocab_size, vocab_tokens,
            token_to_coord, coord_table, ctrl_coord_table, wf_table, ctrl_wf_table,
            lattice, wrouter, uniform_baseline, neighbor_baseline, held_out_set, len(coord_pairs))


def get_batch(split, train_data, val_data, bsz=None):
    bsz = bsz or batch_size
    data = train_data if split == 'train' else val_data
    dx, dy, dc, dcs, dw, dwr = data
    ix = torch.randint(len(dx), (bsz,), device=device)
    return dx[ix], dy[ix], dc[ix], dcs[ix], dw[ix], dwr[ix]


ARM_VANILLA = 0
ARM_MATCHED = 1
ARM_LATTICE = 2
ARM_LATTICE_SHUF = 3
ARM_WORMHOLE = 4
ARM_WORMHOLE_RAND = 5
ARM_FULL = 6
ARM_HYBRID_CONTROL = 7


def arm_forward(model, xb, yb, cb, cbs, wb, wbr, arm_type):
    if arm_type in (ARM_VANILLA, ARM_MATCHED):
        return model(xb, yb)
    if arm_type == ARM_LATTICE:
        return model(xb, yb, coord_indices=cb)
    if arm_type == ARM_LATTICE_SHUF:
        return model(xb, yb, coord_indices=cbs)
    if arm_type == ARM_WORMHOLE:
        return model(xb, yb, wormhole_flags=wb)
    if arm_type == ARM_WORMHOLE_RAND:
        return model(xb, yb, wormhole_flags=wbr)
    if arm_type == ARM_FULL:
        return model(xb, yb, coord_indices=cb, wormhole_flags=wb)
    return model(xb, yb, coord_indices=cbs, wormhole_flags=wbr)


@torch.no_grad()
def evaluate_full_val(model, val_data, val_mask, val_train_mask, arm_type):
    model.eval()
    val_x, val_y, val_c, val_c_shuf, val_wf, val_wf_rand = val_data
    total_loss = 0.0
    total_correct = 0
    total_scored = 0
    edge_loss = 0.0
    edge_correct = 0
    edge_scored = 0
    bsz = batch_size
    n_samples = len(val_x)
    for i in range(0, n_samples, bsz):
        xb = val_x[i:i + bsz]
        yb = val_y[i:i + bsz]
        cb = val_c[i:i + bsz]
        cbs = val_c_shuf[i:i + bsz]
        wb = val_wf[i:i + bsz]
        wbr = val_wf_rand[i:i + bsz]
        mb = val_mask[i:i + bsz]
        tb = val_train_mask[i:i + bsz]
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            logits, _ = arm_forward(model, xb, None, cb, cbs, wb, wbr, arm_type)
        if mb.any():
            scored_logits = logits[mb]
            scored_targets = yb[mb]
            loss = F.cross_entropy(scored_logits, scored_targets, reduction='sum')
            total_loss += loss.item()
            preds = scored_logits.argmax(dim=-1)
            total_correct += (preds == scored_targets).sum().item()
            total_scored += len(scored_targets)
        if tb.any():
            edge_logits = logits[tb]
            edge_targets = yb[tb]
            edge_loss += F.cross_entropy(edge_logits, edge_targets, reduction='sum').item()
            edge_correct += (edge_logits.argmax(dim=-1) == edge_targets).sum().item()
            edge_scored += len(edge_targets)
    model.train()
    mean_val_loss = total_loss / max(1, total_scored)
    val_acc = total_correct / max(1, total_scored)
    mean_edge_loss = edge_loss / max(1, edge_scored)
    train_edge_acc = edge_correct / max(1, edge_scored)
    return mean_val_loss, val_acc, mean_edge_loss, train_edge_acc


@torch.no_grad()
def estimate_progress_train(model, train_data, val_data, arm_type, n_batches=None):
    n_batches = n_batches or eval_iters
    model.eval()
    tr_total = 0.0
    for _ in range(n_batches):
        xb, yb, cb, cbs, wb, wbr = get_batch('train', train_data, val_data)
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            _, l = arm_forward(model, xb, yb, cb, cbs, wb, wbr, arm_type)
        tr_total += l.item()
    model.train()
    return tr_total / n_batches


def train_model(model, arm_name, seed, train_data, val_data, val_mask, val_train_mask,
                iters=None, arm_type=ARM_VANILLA):
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

    final_train_loss = 0.0
    for step in range(iters):
        xb, yb, cb, cbs, wb, wbr = get_batch('train', train_data, val_data)
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            _, loss = arm_forward(compiled_model, xb, yb, cb, cbs, wb, wbr, arm_type)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        lr = get_lr(step, iters, learning_rate, warmup_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if (step + 1) % eval_interval == 0:
            step_tr = estimate_progress_train(model, train_data, val_data, arm_type)
            step_vl, _, _, _ = evaluate_full_val(model, val_data, val_mask, val_train_mask, arm_type)
            final_train_loss = step_tr
            print(f"{arm_name}, seed {seed}, step {step + 1}: "
                  f"train loss {step_tr:.4f}, val loss {step_vl:.4f}")

    if final_train_loss == 0.0:
        final_train_loss = estimate_progress_train(model, train_data, val_data, arm_type)
    final_val_loss, final_val_acc, final_edge_loss, final_edge_acc = evaluate_full_val(
        model, val_data, val_mask, val_train_mask, arm_type)
    return final_val_loss, final_val_acc, final_edge_loss, final_edge_acc, final_train_loss


init_data = generate_seed_data(lattice_dims, seq_len, num_seqs, seed=42)
(init_tr, init_va, init_vm, init_vtm, vocab_size, vocab_tokens,
 token_to_coord, coord_table, ctrl_coord_table, wf_table, ctrl_wf_table,
 lattice_initial, wrouter_initial, _, _, held_out_set_initial, n_edges_total) = init_data


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

all_codes = list(lattice_initial.nodes.keys())
passed = True
for code, coords in list(lattice_initial.nodes.items())[:50]:
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
        pd = lattice_initial.manhattan_distance(sampled[i], sampled[j])
        wd = wrouter_initial.shortest_path_length(sampled[i], sampled[j])
        plain_dists.append(pd)
        worm_dists.append(wd)
avg_plain = sum(plain_dists) / len(plain_dists)
avg_worm = sum(worm_dists) / len(worm_dists)
print(f"Wormhole path length (graph property) - plain: {avg_plain:.2f}, wormholes: {avg_worm:.2f}")


def make_vanilla():
    return GPTLanguageModel(vocab_size, embd_dim=exp_embd, num_heads=exp_heads,
                            num_layers=exp_layers, blk_size=exp_block, drop=exp_drop).to(device)


def make_matched():
    return GPTLanguageModel(vocab_size, embd_dim=matched_embd, num_heads=exp_heads,
                            num_layers=matched_layers, blk_size=exp_block, drop=exp_drop).to(device)


def make_lattice():
    return DASAugmentedGPT(vocab_size, lattice_dims, embd_dim=exp_embd, num_heads=exp_heads,
                           num_layers=exp_layers, blk_size=exp_block, drop=exp_drop,
                           use_lattice_emb=True, use_wormhole_feat=False).to(device)


def make_wormhole():
    return DASAugmentedGPT(vocab_size, lattice_dims, embd_dim=exp_embd, num_heads=exp_heads,
                           num_layers=exp_layers, blk_size=exp_block, drop=exp_drop,
                           use_lattice_emb=False, use_wormhole_feat=True).to(device)


def make_full():
    return DASAugmentedGPT(vocab_size, lattice_dims, embd_dim=exp_embd, num_heads=exp_heads,
                           num_layers=exp_layers, blk_size=exp_block, drop=exp_drop,
                           use_lattice_emb=True, use_wormhole_feat=True).to(device)


arm_specs = [
    ("Vanilla", make_vanilla, ARM_VANILLA),
    ("Vanilla-Matched", make_matched, ARM_MATCHED),
    ("Vanilla+Lattice", make_lattice, ARM_LATTICE),
    ("Lattice-Shuffled", make_lattice, ARM_LATTICE_SHUF),
    ("Vanilla+Wormhole", make_wormhole, ARM_WORMHOLE),
    ("Wormhole-Random", make_wormhole, ARM_WORMHOLE_RAND),
    ("Full-Hybrid", make_full, ARM_FULL),
    ("Hybrid-Control", make_full, ARM_HYBRID_CONTROL),
]

results = {}
for arm_name, _, _ in arm_specs:
    results[arm_name] = {'val_losses': [], 'bpcs': [], 'val_accs': [],
                         'edge_losses': [], 'edge_accs': [], 'train_losses': [], 'params': 0}

saved_models = {}
seed_0_val = None
seed_0_mask = None
seed_0_train_mask = None
oracle_floors = []
uniform_baselines = []

for s_idx, seed in enumerate(seeds):
    s_data = generate_seed_data(lattice_dims, seq_len, num_seqs, seed=seed)
    (tr_d, va_d, va_m, va_tm, *_, s_uniform, s_neighbor, _, _) = s_data
    uniform_baselines.append(s_uniform)
    oracle_floors.append(s_neighbor)

    if s_idx == 0:
        seed_0_val = va_d
        seed_0_mask = va_m
        seed_0_train_mask = va_tm

    for arm_name, make_fn, arm_type in arm_specs:
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
        model = make_fn()
        results[arm_name]['params'] = count_params(model)
        vl, vacc, el, eacc, tl = train_model(model, arm_name, seed, tr_d, va_d, va_m, va_tm,
                                              arm_type=arm_type)
        results[arm_name]['val_losses'].append(vl)
        results[arm_name]['bpcs'].append(vl / math.log(2))
        results[arm_name]['val_accs'].append(vacc)
        results[arm_name]['edge_losses'].append(el)
        results[arm_name]['edge_accs'].append(eacc)
        results[arm_name]['train_losses'].append(tl)

        if s_idx == 0:
            saved_models[arm_name] = model
        else:
            del model
        if use_cuda:
            torch.cuda.empty_cache()


def eval_conformal(model, val_data, val_mask, val_train_mask, arm_type, alpha=0.1, min_test=1000):
    model.eval()
    val_x, val_y, val_c, val_c_shuf, val_wf, val_wf_rand = val_data
    all_logits = []
    all_targets = []
    bsz = batch_size
    for i in range(0, len(val_x), bsz):
        xb = val_x[i:i + bsz]
        yb = val_y[i:i + bsz]
        cb = val_c[i:i + bsz]
        cbs = val_c_shuf[i:i + bsz]
        wb = val_wf[i:i + bsz]
        wbr = val_wf_rand[i:i + bsz]
        mb = val_mask[i:i + bsz] | val_train_mask[i:i + bsz]
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            logits, _ = arm_forward(model, xb, None, cb, cbs, wb, wbr, arm_type)
        if mb.any():
            all_logits.append(logits[mb].detach().cpu())
            all_targets.append(yb[mb].detach().cpu())
    scored_logits = torch.cat(all_logits, dim=0)
    scored_targets = torch.cat(all_targets, dim=0)
    probs = F.softmax(scored_logits, dim=-1)
    true_probs = probs[torch.arange(len(scored_targets)), scored_targets]
    scores = (1.0 - true_probs).tolist()

    order = list(range(len(scores)))
    random.Random(7).shuffle(order)
    n_test = min(len(order), max(min_test, len(order) // 2))
    n_cal = len(order) - n_test
    cal_idx = order[:n_cal]
    test_idx = order[n_cal:]

    conf = ConformalRouter(alpha=alpha)
    conf.calibrate([scores[i] for i in cal_idx])
    thresh = conf.get_threshold()
    emp_cov = conf.coverage([scores[i] for i in test_idx])
    test_probs = probs[test_idx]
    set_sizes = (test_probs >= (1.0 - thresh)).sum(dim=-1).float()
    avg_size = set_sizes.mean().item()
    return 1.0 - alpha, emp_cov, avg_size, n_cal, n_test


alpha = 0.1
for conformal_name, conformal_type in (("Full-Hybrid", ARM_FULL), ("Vanilla", ARM_VANILLA)):
    target_cov, emp_cov, avg_size, n_cal, n_test = eval_conformal(
        saved_models[conformal_name], seed_0_val, seed_0_mask, seed_0_train_mask,
        conformal_type, alpha)
    print(f"Conformal {conformal_name} - target: {target_cov:.2f}, empirical: {emp_cov:.4f}, "
          f"avg set size: {avg_size:.2f}, calibration positions: {n_cal}, test positions: {n_test}")


def measure_generation(model, arm_type, token_to_coord, coord_table, ctrl_coord_table,
                       wf_table, ctrl_wf_table, held_out_set, n_steps=200, n_chains=5, seed=42):
    model.eval()
    use_coord = arm_type in (ARM_LATTICE, ARM_LATTICE_SHUF, ARM_FULL, ARM_HYBRID_CONTROL)
    use_flags = arm_type in (ARM_WORMHOLE, ARM_WORMHOLE_RAND, ARM_FULL, ARM_HYBRID_CONTROL)
    use_shuffled = arm_type in (ARM_LATTICE_SHUF, ARM_HYBRID_CONTROL)
    use_random_flags = arm_type in (ARM_WORMHOLE_RAND, ARM_HYBRID_CONTROL)

    coords_lookup = ctrl_coord_table if use_shuffled else coord_table
    flags_lookup = ctrl_wf_table if use_random_flags else wf_table
    noise_token = len(token_to_coord) - 1

    valid_moves = 0
    total_node_moves = 0
    held_out_moves = 0
    with torch.no_grad():
        for chain in range(n_chains):
            torch.manual_seed(seed + chain)
            start_tok = chain % noise_token
            idx = torch.tensor([[start_tok]], dtype=torch.long, device=device)
            for _ in range(n_steps):
                idx_cond = idx[:, -model.blk_size:]
                ci = coords_lookup[idx_cond] if use_coord else None
                wf = flags_lookup[idx_cond] if use_flags else None
                if arm_type in (ARM_VANILLA, ARM_MATCHED):
                    logits, _ = model(idx_cond)
                else:
                    logits, _ = model(idx_cond, coord_indices=ci, wormhole_flags=wf)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, nxt), dim=1)

            gen = idx[0].tolist()
            for t in range(len(gen) - 1):
                u, v = gen[t], gen[t + 1]
                if u != noise_token and v != noise_token:
                    c1 = token_to_coord[u]
                    c2 = token_to_coord[v]
                    total_node_moves += 1
                    if sum(abs(a - b) for a, b in zip(c1, c2)) == 1:
                        valid_moves += 1
                        if (c1, c2) in held_out_set:
                            held_out_moves += 1
    valid_frac = valid_moves / max(1, total_node_moves)
    held_frac = held_out_moves / max(1, valid_moves)
    return valid_frac, held_frac, total_node_moves, valid_moves


natural_held_out = init_vm.sum().item() / max(1, (init_vm | init_vtm).sum().item())
print(f"Natural held-out transition rate in validation split: {natural_held_out:.4f}")
print(f"Held-out edges: {len(held_out_set_initial)} of {n_edges_total} lattice edges")
print(f"Generation metrics use the seed {seeds[0]} model of each arm")

gen_results = {}
for arm_name, _, arm_type in arm_specs:
    vfrac, hfrac, total_moves, valid_moves = measure_generation(
        saved_models[arm_name], arm_type, token_to_coord, coord_table,
        ctrl_coord_table, wf_table, ctrl_wf_table, held_out_set_initial, 200, 5)
    gen_results[arm_name] = (vfrac, hfrac)
    print(f"Valid lattice move fraction - {arm_name}: {vfrac:.4f} ({valid_moves}/{total_moves})")
    print(f"Generated held-out-edge fraction - {arm_name}: {hfrac:.4f} "
          f"(held-out edges among generated valid moves)")

for m in saved_models.values():
    del m
del saved_models
if use_cuda:
    torch.cuda.empty_cache()

overall_oracle = sum(oracle_floors) / len(oracle_floors)
overall_uniform = sum(uniform_baselines) / len(uniform_baselines)

header = (f"{'Arm':<20} {'Params':>8} {'Val Loss':>19} {'BPC':>19} "
          f"{'Held-out Acc':>19} {'Train-Edge Loss':>19} {'Train-Edge Acc':>19} "
          f"{'Train Loss':>19} {'Valid Move':>11} {'Gen Held-Out':>13}")
print(header)
print("-" * 185)

for arm_name, _, _ in arm_specs:
    r = results[arm_name]
    vl_m, vl_s = mean_std(r['val_losses'])
    bpc_m, bpc_s = mean_std(r['bpcs'])
    acc_m, acc_s = mean_std(r['val_accs'])
    el_m, el_s = mean_std(r['edge_losses'])
    edge_m, edge_s = mean_std(r['edge_accs'])
    tl_m, tl_s = mean_std(r['train_losses'])
    params = r['params']
    vf, hf = gen_results[arm_name]
    print(f"{arm_name:<20} {params:>8,} "
          f"{vl_m:>8.4f} +/- {vl_s:.4f} "
          f"{bpc_m:>8.4f} +/- {bpc_s:.4f} "
          f"{acc_m:>8.4f} +/- {acc_s:.4f} "
          f"{el_m:>8.4f} +/- {el_s:.4f} "
          f"{edge_m:>8.4f} +/- {edge_s:.4f} "
          f"{tl_m:>8.4f} +/- {tl_s:.4f} "
          f"{vf:>11.4f} {hf:>13.4f}")

print(f"Uniform baseline (log V): {overall_uniform:.4f}, "
      f"Neighbor baseline (oracle floor): {overall_oracle:.4f}")

for key_arm in ("Full-Hybrid", "Hybrid-Control"):
    r = results[key_arm]
    for i, seed in enumerate(seeds):
        print(f"{key_arm}, seed {seed}: held-out val loss {r['val_losses'][i]:.4f}, "
              f"train-edge loss {r['edge_losses'][i]:.4f}, "
              f"train loss {r['train_losses'][i]:.4f}, "
              f"held-out acc {r['val_accs'][i]:.4f}, train-edge acc {r['edge_accs'][i]:.4f}")

diffs = [results['Full-Hybrid']['val_losses'][i] - results['Hybrid-Control']['val_losses'][i]
         for i in range(len(seeds))]
diff_m, diff_s = mean_std(diffs)
print(f"Full-Hybrid minus Hybrid-Control held-out val loss: {diff_m:.4f} +/- {diff_s:.4f}")

with open('john_wick.txt', 'r', encoding='utf-8') as f:
    jw_text = f.read()

jw_chars = sorted(list(set(jw_text)))
jw_stoi = {ch: i for i, ch in enumerate(jw_chars)}
jw_itos = {i: ch for i, ch in enumerate(jw_chars)}
jw_data = torch.tensor([jw_stoi[c] for c in jw_text], dtype=torch.long, device=device)
n_jw = int(0.9 * len(jw_data))
train_jw = jw_data[:n_jw]
val_jw = jw_data[n_jw:]

jw_model = GPTLanguageModel(len(jw_chars), embd_dim=exp_embd, num_heads=exp_heads,
                            num_layers=exp_layers, blk_size=exp_block, drop=exp_drop).to(device)
jw_opt = torch.optim.AdamW(jw_model.parameters(), lr=learning_rate)
jw_iters = 3000 if use_cuda else 1500

for step in range(jw_iters):
    ix = torch.randint(len(train_jw) - exp_block - 1, (batch_size,), device=device)
    xb = torch.stack([train_jw[i:i + exp_block] for i in ix])
    yb = torch.stack([train_jw[i + 1:i + exp_block + 1] for i in ix])
    with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                        dtype=amp_dtype, enabled=amp_enabled):
        _, loss = jw_model(xb, yb)
    jw_opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(jw_model.parameters(), 1.0)
    jw_opt.step()

    lr = get_lr(step, jw_iters, learning_rate, warmup_steps)
    for pg in jw_opt.param_groups:
        pg['lr'] = lr

    if (step + 1) % 500 == 0:
        jw_model.eval()
        with torch.no_grad():
            tr_losses = []
            for _ in range(5):
                ix_t = torch.randint(len(train_jw) - exp_block - 1, (batch_size,), device=device)
                xt = torch.stack([train_jw[i:i + exp_block] for i in ix_t])
                yt = torch.stack([train_jw[i + 1:i + exp_block + 1] for i in ix_t])
                with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                                    dtype=amp_dtype, enabled=amp_enabled):
                    _, l_tr = jw_model(xt, yt)
                tr_losses.append(l_tr.item())
            va_losses = []
            for _ in range(5):
                ix_v = torch.randint(len(val_jw) - exp_block - 1, (batch_size,), device=device)
                xv = torch.stack([val_jw[i:i + exp_block] for i in ix_v])
                yv = torch.stack([val_jw[i + 1:i + exp_block + 1] for i in ix_v])
                with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                                    dtype=amp_dtype, enabled=amp_enabled):
                    _, l_va = jw_model(xv, yv)
                va_losses.append(l_va.item())
        jw_model.train()
        mean_tr = sum(tr_losses) / len(tr_losses)
        mean_va = sum(va_losses) / len(va_losses)
        print(f"John-Wick, step {step + 1}: train loss {mean_tr:.4f}, val loss {mean_va:.4f}")

jw_model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    out_tokens = jw_model.generate(context, max_new_tokens=500)[0].tolist()
print(''.join(jw_itos.get(t, '') for t in out_tokens).strip())
del jw_model

print(f"Total run time: {(time.time() - start_time) / 60:.1f} minutes")
