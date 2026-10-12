import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import itertools
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    block_size = 256
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
else:
    block_size = 64
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.0


def morton_encode_4d(x, y, z, w):
    result = 0
    for i in range(16):
        result |= ((x >> i) & 1) << (4 * i)
        result |= ((y >> i) & 1) << (4 * i + 1)
        result |= ((z >> i) & 1) << (4 * i + 2)
        result |= ((w >> i) & 1) << (4 * i + 3)
    return result


def morton_decode_4d(code):
    x = y = z = w = 0
    for i in range(16):
        x |= ((code >> (4 * i)) & 1) << i
        y |= ((code >> (4 * i + 1)) & 1) << i
        z |= ((code >> (4 * i + 2)) & 1) << i
        w |= ((code >> (4 * i + 3)) & 1) << i
    return x, y, z, w


class VoxelLattice:
    def __init__(self, dims):
        self.dims = dims
        self.nodes = {}
        self.adjacency = {}
        for coords in itertools.product(*(range(d) for d in dims)):
            code = morton_encode_4d(*coords)
            self.nodes[code] = coords
            self.adjacency[code] = []
        for code, coords in self.nodes.items():
            for axis in range(4):
                for delta in (-1, 1):
                    neighbor = list(coords)
                    neighbor[axis] += delta
                    if 0 <= neighbor[axis] < self.dims[axis]:
                        nc = morton_encode_4d(*neighbor)
                        self.adjacency[code].append(nc)

    def get_coords(self, code):
        return self.nodes.get(code, None)

    def get_code(self, coords):
        return morton_encode_4d(*coords)

    def shortest_path_length(self, src, dst):
        if src == dst:
            return 0
        visited = {src}
        frontier = [src]
        dist = 0
        while frontier:
            dist += 1
            next_frontier = []
            for node in frontier:
                for nb in self.adjacency[node]:
                    if nb == dst:
                        return dist
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
            frontier = next_frontier
        return float('inf')

    def manhattan_distance(self, src, dst):
        c1 = self.nodes[src]
        c2 = self.nodes[dst]
        return sum(abs(a - b) for a, b in zip(c1, c2))


class WormholeRouter:
    def __init__(self, lattice, num_wormholes=None, seed=42):
        self.lattice = lattice
        codes = list(lattice.nodes.keys())
        rng = random.Random(seed)
        if num_wormholes is None:
            num_wormholes = max(1, len(codes) // 4)
        self.wormholes = {}
        candidates = codes[:]
        rng.shuffle(candidates)
        for i in range(0, min(num_wormholes * 2, len(candidates)) - 1, 2):
            a, b = candidates[i], candidates[i + 1]
            self.wormholes[a] = b
            self.wormholes[b] = a
        self.augmented_adjacency = {}
        for code in codes:
            adj = list(lattice.adjacency[code])
            if code in self.wormholes:
                adj.append(self.wormholes[code])
            self.augmented_adjacency[code] = adj

    def shortest_path_length(self, src, dst):
        if src == dst:
            return 0
        visited = {src}
        frontier = [src]
        dist = 0
        while frontier:
            dist += 1
            next_frontier = []
            for node in frontier:
                for nb in self.augmented_adjacency[node]:
                    if nb == dst:
                        return dist
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
            frontier = next_frontier
        return float('inf')

    def has_wormhole(self, code):
        return code in self.wormholes

    def wormhole_target(self, code):
        return self.wormholes.get(code, None)


class ConformalRouter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.calibration_scores = []

    def calibrate(self, scores):
        self.calibration_scores = sorted(scores)

    def get_threshold(self):
        n = len(self.calibration_scores)
        if n == 0:
            return float('inf')
        q_level = math.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        idx = int(math.ceil(q_level * n)) - 1
        idx = max(0, min(idx, n - 1))
        return self.calibration_scores[idx]

    def coverage(self, test_true_scores):
        threshold = self.get_threshold()
        if not test_true_scores:
            return 0.0
        covered = sum(1 for s in test_true_scores if s <= threshold)
        return covered / len(test_true_scores)


class CausalSelfAttention(nn.Module):
    def __init__(self, embd_dim, num_heads, drop):
        super().__init__()
        self.n_head = num_heads
        self.head_dim = embd_dim // num_heads
        self.c_attn = nn.Linear(embd_dim, 3 * embd_dim, bias=False)
        self.c_proj = nn.Linear(embd_dim, embd_dim, bias=False)
        self.attn_dropout = drop
        self.resid_drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        drop_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.c_proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, embd_dim, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.GELU(),
            nn.Linear(4 * embd_dim, embd_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embd_dim, num_heads, drop):
        super().__init__()
        self.sa = CausalSelfAttention(embd_dim, num_heads, drop)
        self.ffwd = FeedForward(embd_dim, drop)
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embd_dim=None, num_heads=None, num_layers=None,
                 blk_size=None, drop=None):
        super().__init__()
        self.embd_dim = embd_dim or n_embd
        self.num_heads = num_heads or n_head
        self.num_layers = num_layers or n_layer
        self.blk_size = blk_size or block_size
        self.drop = drop if drop is not None else dropout
        self.token_embedding_table = nn.Embedding(vocab_size, self.embd_dim)
        self.position_embedding_table = nn.Embedding(self.blk_size, self.embd_dim)
        self.blocks = nn.Sequential(
            *[Block(self.embd_dim, self.num_heads, self.drop) for _ in range(self.num_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embd_dim)
        self.lm_head = nn.Linear(self.embd_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.blk_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class LatticeEmbedding(nn.Module):
    def __init__(self, dims, embd_dim):
        super().__init__()
        self.dims = dims
        per_axis = embd_dim // 4
        self.coord_embeds = nn.ModuleList([
            nn.Embedding(d, per_axis) for d in dims
        ])
        total = per_axis * 4
        self.proj = nn.Linear(total, embd_dim) if total != embd_dim else nn.Identity()

    def forward(self, coord_indices):
        parts = []
        for i in range(4):
            parts.append(self.coord_embeds[i](coord_indices[:, :, i]))
        out = torch.cat(parts, dim=-1)
        out = self.proj(out)
        return out


class WormholeFeature(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.wormhole_flag_emb = nn.Embedding(2, embd_dim)
        self.gate = nn.Linear(embd_dim * 2, embd_dim)

    def forward(self, x, wormhole_flags):
        wf = self.wormhole_flag_emb(wormhole_flags)
        combined = torch.cat([x, wf], dim=-1)
        return self.gate(combined)


class DASAugmentedGPT(nn.Module):
    def __init__(self, vocab_size, lattice_dims, embd_dim=None, num_heads=None,
                 num_layers=None, blk_size=None, drop=None,
                 use_lattice_emb=False, use_wormhole_feat=False):
        super().__init__()
        self.embd_dim = embd_dim or n_embd
        self.num_heads = num_heads or n_head
        self.num_layers = num_layers or n_layer
        self.blk_size = blk_size or block_size
        self.drop = drop if drop is not None else dropout
        self.use_lattice_emb = use_lattice_emb
        self.use_wormhole_feat = use_wormhole_feat
        self.token_embedding_table = nn.Embedding(vocab_size, self.embd_dim)
        self.position_embedding_table = nn.Embedding(self.blk_size, self.embd_dim)
        if self.use_lattice_emb:
            self.lattice_emb = LatticeEmbedding(lattice_dims, self.embd_dim)
            self.lattice_gate = nn.Linear(self.embd_dim * 2, self.embd_dim)
        if self.use_wormhole_feat:
            self.wormhole_feat = WormholeFeature(self.embd_dim)
        self.blocks = nn.Sequential(
            *[Block(self.embd_dim, self.num_heads, self.drop) for _ in range(self.num_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embd_dim)
        self.lm_head = nn.Linear(self.embd_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, coord_indices=None, wormhole_flags=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        if self.use_lattice_emb and coord_indices is not None:
            lat_emb = self.lattice_emb(coord_indices)
            gate_input = torch.cat([x, lat_emb], dim=-1)
            x = self.lattice_gate(gate_input)
        if self.use_wormhole_feat and wormhole_flags is not None:
            x = self.wormhole_feat(x, wormhole_flags)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens, coord_indices=None, wormhole_flags=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.blk_size:]
            ci = coord_indices[:, -self.blk_size:] if coord_indices is not None else None
            wf = wormhole_flags[:, -self.blk_size:] if wormhole_flags is not None else None
            logits, _ = self(idx_cond, coord_indices=ci, wormhole_flags=wf)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
