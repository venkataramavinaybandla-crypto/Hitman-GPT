import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random
import itertools
import sys
import time
import json
import os
import webbrowser

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

split_seeds = [1, 2, 3]
train_seeds = [42, 137, 2024]
seeds = train_seeds

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

    for _s in train_seqs:
        for _t in range(len(_s) - 1):
            if _s[_t] != noise_token and _s[_t + 1] != noise_token:
                assert (token_to_coord[_s[_t]], token_to_coord[_s[_t + 1]]) not in held_out_set

    tr_loops = sum(s[t] == s[t + 1] for s in train_seqs for t in range(len(s) - 1) if s[t] != noise_token and s[t + 1] != noise_token)
    tr_trans = sum(1 for s in train_seqs for t in range(len(s) - 1) if s[t] != noise_token and s[t + 1] != noise_token)
    va_loops = sum(s[t] == s[t + 1] for s in val_seqs for t in range(len(s) - 1) if s[t] != noise_token and s[t + 1] != noise_token)
    va_trans = sum(1 for s in val_seqs for t in range(len(s) - 1) if s[t] != noise_token and s[t + 1] != noise_token)
    print(f"Self-loop fraction - train: {tr_loops / max(1, tr_trans):.4f}, val: {va_loops / max(1, va_trans):.4f}")

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
    held_coords = [token_to_coord[val_seqs[i][t]] for i in range(len(val_seqs)) for t in range(seq_length - 1) if val_mask[i, t].item()]
    held_entropy = sum(math.log(len(lattice_adj[c])) for c in held_coords) / max(1, len(held_coords))
    held_oracle = held_entropy - math.log(1 - noise_rate)
    edge_coords = [token_to_coord[val_seqs[i][t]] for i in range(len(val_seqs)) for t in range(seq_length - 1) if val_train_mask[i, t].item()]
    edge_entropy = sum(math.log(len(lattice_adj[c])) for c in edge_coords) / max(1, len(edge_coords))
    edge_oracle = edge_entropy - math.log(1 - noise_rate)

    train_data = (train_x, train_y, train_c, train_c_shuf, train_wf, train_wf_rand)
    val_data = (val_x, val_y, val_c, val_c_shuf, val_wf, val_wf_rand)

    return (train_data, val_data, val_mask, val_train_mask, vocab_size, vocab_tokens,
            token_to_coord, coord_table, ctrl_coord_table, wf_table, ctrl_wf_table,
            lattice, wrouter, uniform_baseline, (held_entropy, held_oracle, edge_entropy, edge_oracle),
            held_out_set, len(coord_pairs))


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
    eval_history = []
    for step in range(iters):
        xb, yb, cb, cbs, wb, wbr = get_batch('train', train_data, val_data)
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            _, loss = arm_forward(compiled_model, xb, yb, cb, cbs, wb, wbr, arm_type)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, iters, learning_rate, warmup_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if (step + 1) % eval_interval == 0:
            step_tr = estimate_progress_train(model, train_data, val_data, arm_type)
            step_vl, _, _, _ = evaluate_full_val(model, val_data, val_mask, val_train_mask, arm_type)
            final_train_loss = step_tr
            eval_history.append({'step': step + 1, 'train_loss': round(step_tr, 4), 'val_loss': round(step_vl, 4)})
            print(f"{arm_name}, seed {seed}, step {step + 1}: "
                  f"train loss {step_tr:.4f}, val loss {step_vl:.4f}")

    if final_train_loss == 0.0:
        final_train_loss = estimate_progress_train(model, train_data, val_data, arm_type)
    final_val_loss, final_val_acc, final_edge_loss, final_edge_acc = evaluate_full_val(
        model, val_data, val_mask, val_train_mask, arm_type)
    return final_val_loss, final_val_acc, final_edge_loss, final_edge_acc, final_train_loss, eval_history


init_data = generate_seed_data(lattice_dims, seq_len, num_seqs, seed=split_seeds[0])
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
held_entropies = []
held_oracles = []
edge_entropies = []
edge_oracles = []
uniform_baselines = []

run_idx = 0
for s_idx, split_seed in enumerate(split_seeds):
    s_data = generate_seed_data(lattice_dims, seq_len, num_seqs, seed=split_seed)
    (tr_d, va_d, va_m, va_tm, *_, s_uniform, s_baselines, _, _) = s_data
    uniform_baselines.append(s_uniform)
    held_entropies.append(s_baselines[0])
    held_oracles.append(s_baselines[1])
    edge_entropies.append(s_baselines[2])
    edge_oracles.append(s_baselines[3])

    if s_idx == 0:
        seed_0_val = va_d
        seed_0_mask = va_m
        seed_0_train_mask = va_tm

    for train_seed in train_seeds:
        for arm_name, make_fn, arm_type in arm_specs:
            torch.manual_seed(train_seed)
            if use_cuda:
                torch.cuda.manual_seed(train_seed)
            model = make_fn()
            results[arm_name]['params'] = count_params(model)
            vl, vacc, el, eacc, tl, eh = train_model(model, arm_name, train_seed, tr_d, va_d, va_m, va_tm,
                                                  arm_type=arm_type)
            results[arm_name]['val_losses'].append(vl)
            results[arm_name]['bpcs'].append(vl / math.log(2))
            results[arm_name]['val_accs'].append(vacc)
            results[arm_name]['edge_losses'].append(el)
            results[arm_name]['edge_accs'].append(eacc)
            results[arm_name]['train_losses'].append(tl)
            if run_idx == 0:
                results[arm_name]['eval_history'] = eh
            print(f"Run (split {split_seed}, train {train_seed}) - {arm_name}: held-out loss {vl:.4f}, held-out acc {vacc:.4f}, train-edge loss {el:.4f}, train-edge acc {eacc:.4f}, train loss {tl:.4f}")

            if run_idx == 0:
                saved_models[arm_name] = model
            else:
                del model
            if use_cuda:
                torch.cuda.empty_cache()
        run_idx += 1


def eval_conformal(model, val_data, val_mask, val_train_mask, arm_type, alpha=0.1, seed=42):
    model.eval()
    val_x, val_y, val_c, val_c_shuf, val_wf, val_wf_rand = val_data
    n_seqs = len(val_x)
    seq_order = list(range(n_seqs))
    random.Random(seed).shuffle(seq_order)
    cal_seq_set = set(seq_order[:n_seqs // 2])

    bsz = batch_size
    cal_held_scores, test_held_scores = [], []
    cal_edge_scores, test_edge_scores = [], []
    test_held_probs, test_edge_probs = [], []

    for i in range(0, n_seqs, bsz):
        xb = val_x[i:i + bsz]
        yb = val_y[i:i + bsz]
        cb = val_c[i:i + bsz]
        cbs = val_c_shuf[i:i + bsz]
        wb = val_wf[i:i + bsz]
        wbr = val_wf_rand[i:i + bsz]
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            logits, _ = arm_forward(model, xb, None, cb, cbs, wb, wbr, arm_type)
        probs = F.softmax(logits, dim=-1)
        true_probs = probs.gather(-1, yb.unsqueeze(-1)).squeeze(-1)
        scores = 1.0 - true_probs

        for b_idx in range(len(xb)):
            seq_id = i + b_idx
            is_cal = seq_id in cal_seq_set
            hm = val_mask[seq_id]
            em = val_train_mask[seq_id]
            if is_cal:
                if hm.any():
                    cal_held_scores.extend(scores[b_idx][hm].tolist())
                if em.any():
                    cal_edge_scores.extend(scores[b_idx][em].tolist())
            else:
                if hm.any():
                    test_held_scores.extend(scores[b_idx][hm].tolist())
                    test_held_probs.append(probs[b_idx][hm].detach().cpu())
                if em.any():
                    test_edge_scores.extend(scores[b_idx][em].tolist())
                    test_edge_probs.append(probs[b_idx][em].detach().cpu())

    results_out = {}
    groups = [
        ('held-out-mask', cal_held_scores, test_held_scores, test_held_probs),
        ('train-edge-mask', cal_edge_scores, test_edge_scores, test_edge_probs),
    ]
    for grp_name, cal_s, test_s, test_p in groups:
        n_cal = len(cal_s)
        n_test = len(test_s)
        if n_cal < 200 or n_test < 200:
            cov_str = "WARNING: under 200 positions"
            size_str = "WARNING: under 200 positions"
        else:
            conf = ConformalRouter(alpha=alpha)
            conf.calibrate(cal_s)
            thresh = conf.get_threshold()
            cov_val = conf.coverage(test_s)
            p_cat = torch.cat(test_p, dim=0)
            set_sizes = (p_cat >= (1.0 - thresh)).sum(dim=-1).float()
            avg_size = set_sizes.mean().item()
            cov_str = f"{cov_val:.4f}"
            size_str = f"{avg_size:.2f}"
        results_out[grp_name] = (n_cal, n_test, cov_str, size_str)

    return results_out


alpha = 0.1
conformal_results_dict = {}
for conformal_name, conformal_type in (("Full-Hybrid", ARM_FULL), ("Vanilla", ARM_VANILLA)):
    conf_res = eval_conformal(
        saved_models[conformal_name], seed_0_val, seed_0_mask, seed_0_train_mask,
        conformal_type, alpha)
    conformal_results_dict[conformal_name] = conf_res
    for grp_name in ("held-out-mask", "train-edge-mask"):
        n_cal, n_test, cov_val, sz_val = conf_res[grp_name]
        print(f"Conformal {conformal_name} [{grp_name}] - target: {1.0 - alpha:.2f}, "
              f"coverage: {cov_val}, mean set size: {sz_val}, "
              f"calibration positions: {n_cal}, test positions: {n_test}")


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
print(f"Generation metrics use split seed {split_seeds[0]}, train seed {train_seeds[0]} model of each arm")

gen_results = {}
for arm_name, _, arm_type in arm_specs:
    vfrac, hfrac, total_moves, valid_moves = measure_generation(
        saved_models[arm_name], arm_type, token_to_coord, coord_table,
        ctrl_coord_table, wf_table, ctrl_wf_table, held_out_set_initial, 200, 5)
    gen_results[arm_name] = (vfrac, hfrac)
    print(f"Valid lattice move fraction - {arm_name}: {vfrac:.4f} ({valid_moves}/{total_moves})")
    print(f"Generated held-out-edge fraction - {arm_name}: {hfrac:.4f} "
          f"(held-out edges among generated valid moves)")

coord_to_tok_map = {v: k for k, v in token_to_coord.items()}
wormhole_pairs_list = []
seen_wh = set()
for c1_code, c2_code in wrouter_initial.wormholes.items():
    c1 = lattice_initial.nodes[c1_code]
    c2 = lattice_initial.nodes[c2_code]
    t1 = coord_to_tok_map[c1]
    t2 = coord_to_tok_map[c2]
    pair = (min(t1, t2), max(t1, t2))
    if pair not in seen_wh:
        seen_wh.add(pair)
        wormhole_pairs_list.append([pair[0], pair[1]])

held_out_edges_list = []
seen_he = set()
for c1, c2 in held_out_set_initial:
    t1 = coord_to_tok_map[c1]
    t2 = coord_to_tok_map[c2]
    pair = (min(t1, t2), max(t1, t2))
    if pair not in seen_he:
        seen_he.add(pair)
        held_out_edges_list.append([pair[0], pair[1]])

all_tess_edges_list = []
seen_te = set()
for code, neighbors in lattice_initial.adjacency.items():
    c1 = lattice_initial.nodes[code]
    t1 = coord_to_tok_map[c1]
    for n_code in neighbors:
        c2 = lattice_initial.nodes[n_code]
        t2 = coord_to_tok_map[c2]
        pair = (min(t1, t2), max(t1, t2))
        if pair not in seen_te:
            seen_te.add(pair)
            all_tess_edges_list.append([pair[0], pair[1]])

train_edges_list = [p for p in all_tess_edges_list if (min(p[0], p[1]), max(p[0], p[1])) not in seen_he]

s0_vx, s0_vy, s0_vc, s0_vcs, s0_vw, s0_vwr = seed_0_val
n_val_dump = min(20, len(s0_vx))
val_seqs_dump = []
m_fh = saved_models["Full-Hybrid"]
m_hc = saved_models["Hybrid-Control"]
m_fh.eval()
m_hc.eval()

with torch.no_grad():
    for sq_i in range(n_val_dump):
        xb = s0_vx[sq_i:sq_i + 1]
        yb = s0_vy[sq_i:sq_i + 1]
        cb = s0_vc[sq_i:sq_i + 1]
        cbs = s0_vcs[sq_i:sq_i + 1]
        wb = s0_vw[sq_i:sq_i + 1]
        wbr = s0_vwr[sq_i:sq_i + 1]
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            l_fh, _ = arm_forward(m_fh, xb, None, cb, cbs, wb, wbr, ARM_FULL)
            l_hc, _ = arm_forward(m_hc, xb, None, cb, cbs, wb, wbr, ARM_HYBRID_CONTROL)
        p_fh = F.softmax(l_fh[0], dim=-1)
        p_hc = F.softmax(l_hc[0], dim=-1)
        tok_seq = xb[0].tolist()
        tar_seq = yb[0].tolist()
        T_seq = len(tok_seq)
        pos_fh = []
        pos_hc = []
        masks_seq = []
        for t in range(T_seq):
            tgt = tar_seq[t]
            is_h = seed_0_mask[sq_i, t].item()
            is_tr = seed_0_train_mask[sq_i, t].item()
            m_t = "held_out" if is_h else ("train_edge" if is_tr else "noise")
            masks_seq.append(m_t)

            pf = p_fh[t]
            pt_f = pf[tgt].item()
            top5_pf, top5_if = torch.topk(pf, k=5)
            pos_fh.append({
                "loss": round(-math.log(max(1e-12, pt_f)), 4),
                "prob_true": round(pt_f, 4),
                "top5": [{"token": int(top5_if[k]), "coord": list(token_to_coord[top5_if[k].item()]), "prob": round(float(top5_pf[k]), 4)} for k in range(5)]
            })

            ph = p_hc[t]
            pt_h = ph[tgt].item()
            top5_ph, top5_ih = torch.topk(ph, k=5)
            pos_hc.append({
                "loss": round(-math.log(max(1e-12, pt_h)), 4),
                "prob_true": round(pt_h, 4),
                "top5": [{"token": int(top5_ih[k]), "coord": list(token_to_coord[top5_ih[k].item()]), "prob": round(float(top5_ph[k]), 4)} for k in range(5)]
            })

        val_seqs_dump.append({
            "seq_idx": sq_i,
            "tokens": tok_seq,
            "targets": tar_seq,
            "masks": masks_seq,
            "full_hybrid": pos_fh,
            "hybrid_control": pos_hc
        })

results_export = {
    "dims": list(lattice_dims),
    "noise_token_id": len(token_to_coord) - 1,
    "token_to_coord": {str(k): list(v) for k, v in token_to_coord.items()},
    "wormhole_pairs": wormhole_pairs_list,
    "held_out_edges": held_out_edges_list,
    "train_edges": train_edges_list,
    "tesseract_edges": all_tess_edges_list,
    "per_arm_metrics": {
        arm: {
            "params": results[arm]["params"],
            "val_losses": [round(x, 4) for x in results[arm]["val_losses"]],
            "bpcs": [round(x, 4) for x in results[arm]["bpcs"]],
            "val_accs": [round(x, 4) for x in results[arm]["val_accs"]],
            "edge_losses": [round(x, 4) for x in results[arm]["edge_losses"]],
            "edge_accs": [round(x, 4) for x in results[arm]["edge_accs"]],
            "train_losses": [round(x, 4) for x in results[arm]["train_losses"]],
            "eval_history": results[arm].get("eval_history", [])
        }
        for arm, _, _ in arm_specs
    },
    "baselines": {
        "uniform_baseline": round(sum(uniform_baselines) / len(uniform_baselines), 4),
        "held_out": {
            "entropy": round(sum(held_entropies) / len(held_entropies), 4),
            "oracle_floor": round(sum(held_oracles) / len(held_oracles), 4)
        },
        "train_edge": {
            "entropy": round(sum(edge_entropies) / len(edge_entropies), 4),
            "oracle_floor": round(sum(edge_oracles) / len(edge_oracles), 4)
        }
    },
    "conformal_results": {
        arm: {
            grp: {
                "n_cal": conf_res[0],
                "n_test": conf_res[1],
                "coverage": conf_res[2],
                "mean_size": conf_res[3]
            }
            for grp, conf_res in conf_dict.items()
        }
        for arm, conf_dict in conformal_results_dict.items()
    },
    "generation_results": {
        arm: {
            "valid_move_frac": round(gen_results[arm][0], 4),
            "held_out_edge_frac": round(gen_results[arm][1], 4)
        }
        for arm, _, _ in arm_specs
    },
    "val_sequences": val_seqs_dump
}

with open("results.json", "w", encoding="utf-8") as f_json:
    json.dump(results_export, f_json, indent=2)

for m in saved_models.values():
    del m
del saved_models
if use_cuda:
    torch.cuda.empty_cache()

overall_held_entropy = sum(held_entropies) / len(held_entropies)
overall_held_oracle = sum(held_oracles) / len(held_oracles)
overall_edge_entropy = sum(edge_entropies) / len(edge_entropies)
overall_edge_oracle = sum(edge_oracles) / len(edge_oracles)
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

print(f"Uniform baseline (log V): {overall_uniform:.4f}")
print(f"Held-out mask - noise-free neighbour entropy: {overall_held_entropy:.4f}, "
      f"oracle floor: {overall_held_oracle:.4f}")
print(f"Train-edge mask - noise-free neighbour entropy: {overall_edge_entropy:.4f}, "
      f"oracle floor: {overall_edge_oracle:.4f}")

for arm_name, _, _ in arm_specs:
    r = results[arm_name]
    vl_m, _ = mean_std(r['val_losses'])
    el_m, _ = mean_std(r['edge_losses'])
    gap_held = vl_m - overall_held_oracle
    gap_edge = el_m - overall_edge_oracle
    print(f"{arm_name} gap to oracle floor - held-out: {gap_held:.4f}, train-edge: {gap_edge:.4f}")

t_table_95 = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
    6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
    11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
    16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
    21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639, 25: 2.0595,
    26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423
}


def paired_stats(vals_a, vals_b):
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    n = len(diffs)
    m = sum(diffs) / n
    sd = (sum((x - m) ** 2 for x in diffs) / max(1, n - 1)) ** 0.5
    se = sd / (n ** 0.5)
    df = min(30, max(1, n - 1))
    t_crit = t_table_95[df]
    ci = (m - t_crit * se, m + t_crit * se)
    return diffs, m, sd, se, ci


pairs = [
    ("Full-Hybrid", "Hybrid-Control", "Full-Hybrid minus Hybrid-Control"),
    ("Full-Hybrid", "Vanilla-Matched", "Full-Hybrid minus Vanilla-Matched"),
    ("Vanilla+Lattice", "Lattice-Shuffled", "Lattice minus Lattice-Shuffled"),
    ("Vanilla+Wormhole", "Wormhole-Random", "Wormhole minus Wormhole-Random"),
]

all_runs = [(s, t) for s in split_seeds for t in train_seeds]
for arm_name, _, _ in arm_specs:
    r = results[arm_name]
    for (s, t), vl, vacc, el, eacc, tl in zip(all_runs, r['val_losses'], r['val_accs'], r['edge_losses'], r['edge_accs'], r['train_losses']):
        print(f"{arm_name}, split {s}, train {t}: held-out val loss {vl:.4f}, train-edge loss {el:.4f}, train loss {tl:.4f}, held-out acc {vacc:.4f}, train-edge acc {eacc:.4f}")

for arm_a, arm_b, label in pairs:
    loss_diffs, loss_m, loss_sd, loss_se, (loss_cil, loss_ciu) = paired_stats(
        results[arm_a]['val_losses'], results[arm_b]['val_losses']
    )
    acc_diffs, acc_m, acc_sd, acc_se, (acc_cil, acc_ciu) = paired_stats(
        results[arm_a]['val_accs'], results[arm_b]['val_accs']
    )
    for (s, t), dl, da in zip(all_runs, loss_diffs, acc_diffs):
        print(f"{label}, split {s}, train {t}: paired diff held-out loss {dl:.4f}, held-out acc {da:.4f}")
    print(f"{label} - held-out loss mean: {loss_m:.4f}, SD: {loss_sd:.4f}, SE: {loss_se:.4f}, 95% CI: [{loss_cil:.4f}, {loss_ciu:.4f}]")
    print(f"{label} - held-out acc mean: {acc_m:.4f}, SD: {acc_sd:.4f}, SE: {acc_se:.4f}, 95% CI: [{acc_cil:.4f}, {acc_ciu:.4f}]")

with open('john_wick.txt', 'r', encoding='utf-8') as f:
    jw_text = f.read()

jw_chars = sorted(list(set(jw_text)))
jw_stoi = {ch: i for i, ch in enumerate(jw_chars)}
jw_itos = {i: ch for i, ch in enumerate(jw_chars)}
jw_data = torch.tensor([jw_stoi[c] for c in jw_text], dtype=torch.long, device=device)
n_jw = int(0.9 * len(jw_data))
train_jw = jw_data[:n_jw]
val_jw = jw_data[n_jw:]

jw_model = GPTLanguageModel(len(jw_chars), embd_dim=n_embd, num_heads=n_head,
                            num_layers=n_layer, blk_size=block_size, drop=dropout).to(device)
jw_opt = torch.optim.AdamW(jw_model.parameters(), lr=learning_rate)
jw_iters = 5000
jw_warmup = max(1, jw_iters // 20)

for step in range(jw_iters):
    ix = torch.randint(len(train_jw) - block_size - 1, (batch_size,), device=device)
    xb = torch.stack([train_jw[i:i + block_size] for i in ix])
    yb = torch.stack([train_jw[i + 1:i + block_size + 1] for i in ix])
    with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                        dtype=amp_dtype, enabled=amp_enabled):
        _, loss = jw_model(xb, yb)
    jw_opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(jw_model.parameters(), 1.0)
    lr = get_lr(step, jw_iters, learning_rate, jw_warmup)
    for pg in jw_opt.param_groups:
        pg['lr'] = lr
    jw_opt.step()

    if (step + 1) % 500 == 0:
        jw_model.eval()
        with torch.no_grad():
            tr_losses = []
            for _ in range(5):
                ix_t = torch.randint(len(train_jw) - block_size - 1, (batch_size,), device=device)
                xt = torch.stack([train_jw[i:i + block_size] for i in ix_t])
                yt = torch.stack([train_jw[i + 1:i + block_size + 1] for i in ix_t])
                with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                                    dtype=amp_dtype, enabled=amp_enabled):
                    _, l_tr = jw_model(xt, yt)
                tr_losses.append(l_tr.item())
            va_losses = []
            for _ in range(5):
                ix_v = torch.randint(len(val_jw) - block_size - 1, (batch_size,), device=device)
                xv = torch.stack([val_jw[i:i + block_size] for i in ix_v])
                yv = torch.stack([val_jw[i + 1:i + block_size + 1] for i in ix_v])
                with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                                    dtype=amp_dtype, enabled=amp_enabled):
                    _, l_va = jw_model(xv, yv)
                va_losses.append(l_va.item())
        jw_model.train()
        mean_tr = sum(tr_losses) / len(tr_losses)
        mean_va = sum(va_losses) / len(va_losses)
        print(f"John-Wick, step {step + 1}: train loss {mean_tr:.4f}, val loss {mean_va:.4f}")

jw_model.eval()
val_loss_sum = 0.0
val_tokens_count = 0
n_val_chunks = (len(val_jw) - 1) // block_size
with torch.no_grad():
    for c_i in range(0, n_val_chunks, batch_size):
        c_end = min(c_i + batch_size, n_val_chunks)
        xb = torch.stack([val_jw[k * block_size : k * block_size + block_size] for k in range(c_i, c_end)])
        yb = torch.stack([val_jw[k * block_size + 1 : k * block_size + block_size + 1] for k in range(c_i, c_end)])
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                            dtype=amp_dtype, enabled=amp_enabled):
            _, loss = jw_model(xb, yb)
        n_elem = xb.numel()
        val_loss_sum += loss.item() * n_elem
        val_tokens_count += n_elem
jw_val_nats = val_loss_sum / max(1, val_tokens_count)
jw_val_bpc = jw_val_nats / math.log(2)
print(f"John-Wick whole val loss: {jw_val_nats:.4f} nats, {jw_val_bpc:.4f} bits per character")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    out_tokens = jw_model.generate(context, max_new_tokens=500)[0].tolist()
print(''.join(jw_itos.get(t, '') for t in out_tokens).strip())
del jw_model

print(f"Total run time: {(time.time() - start_time) / 60:.1f} minutes")
viz_html_path = os.path.abspath("viz.html")
try:
    webbrowser.open("file:///" + viz_html_path.replace("\\", "/"))
except Exception:
    pass
print(f"Visualization path: {viz_html_path}")
