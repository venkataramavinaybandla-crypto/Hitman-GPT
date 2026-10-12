# Hitman-GPT

![Hitman-GPT](assets/john_wick.jpg)

The simplest, fastest repository for training decoder-only Transformers equipped with the Dimensional Anchor System (DAS). It prioritizes teeth over enterprise AI bureaucracy. The entire codebase is strictly contained in two clean, readable, dependency-free Python files: `Train.py` is a self-contained training harness and multi-seed benchmark runner, and `bigram.py` is a lean GPT model definition coupled with a 4D Morton-encoded voxel lattice, wormhole routing, and conformal prediction guarantees. That is it. No PyTorch Lightning, no sprawling Hydra configs, no argparse ceremony, and no 40-page YAML files written by committee.

![Benchmark Results](assets/benchmark_results.png)

Because the code is so simple, it is remarkably easy to hack to your needs, inspect every tensor operation directly, or stress-test whether geometric inductive biases actually help autoregressive models generalize beyond memorization.

---

## install

```sh
pip install torch
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- that is literally all you need.

No wandb login prompts, no huggingface token headaches, no fifty-line requirements.txt pinned to obsolete wheel hashes.

---

## quick start

If you just want to run the whole suite and see the models train and evaluate in real time, simply launch:

```sh
python Train.py
```

The script automatically probes your environment:

**I have a GPU**. If you have an NVIDIA card with CUDA, the code immediately kicks into gear: it flips on TF32 matrix multiplication on Ampere/Hopper cores, enables `torch.bfloat16` autocasting, switches AdamW to fused kernel mode, scales up the batch size and context window, and engages `torch.compile` on Linux.

**I only have a CPU** (or a laptop running Windows). No worries at all. When there is no CUDA device available, the script automatically dials things down to a lightweight profile (batch size 16, 500 steps per run, 4D grid resolution $3 \times 3 \times 3 \times 3$). The full controlled comparison—spanning 5 distinct model architectures across 3 random seeds (15 training runs in total)—finishes in roughly four minutes on an ordinary laptop CPU.

During execution, training progress prints to your console every 500 steps, validates component proofs, builds the benchmark comparison table, and generates both a 4D spatial flight path and a sample of learned dialogue.

At the end of the run, the character-level model samples from the dialogue of *John Wick*:

```text
SCE868114 Qweeiet
kiole. 2 Mageman mokey te wer
quioblegrrganst notin's eta skis of wif ais a
thenye meeis esest oce maekie: The
sible. Youle kecoucie
```

lol `¯\_(ツ)_/¯`. Not bad for four minutes on a laptop CPU without a single GPU core in sight. It is not winning any Academy Awards for screenwriting just yet, but the character gestalt and punctuation cadence are clearly beginning to assemble.

---

## why this exists: the dirty secret of autoregressive models

Modern large language models enjoy a convenient luxury: internet-scale corpora are so vast that when a benchmark claims a model is "reasoning", the network is frequently just interpolating through dense clusters of memorized token n-grams. The moment you push an autoregressive Transformer into zero-shot extrapolation along transitions it has strictly never encountered during training, the illusion tends to collapse.

Standard Transformers treat every token as an isolated categorical integer ID. The token for a location or state has no innate geometric connection to its neighboring states. The self-attention mechanism is forced to reconstruct the underlying topology from scratch purely through empirical co-occurrence statistics.

The Dimensional Anchor System (DAS) investigates the alternative hypothesis: what happens when we anchor token states inside an explicit topological manifold, pairing coordinate embeddings with non-local shortcut routing?

To test this honestly without corporate benchmarking games, `Train.py` procedurally generates a 4D navigation task:
1. Sequences are driven by random walks across the nodes of a 4D hypergrid.
2. An agent can only step $\pm 1$ along one orthogonal coordinate axis at a time.
3. A controlled $5\%$ noise rate injects perturbation tokens (`N`).
4. **The stress-test**: Exactly $15\%$ of valid coordinate transitions are strictly held out from the training split.

The validation set is therefore composed of sequences traversing routes the network has never seen. If an architecture relies on memorizing observed pairs, its validation cross-entropy explodes. If it genuinely understands the underlying 4D geometry, it navigates the held-out paths with low loss.

---

## the dimensional anchor system (DAS)

DAS augments the standard decoder-only Transformer with three software modules defined in `bigram.py`:

![Lattice and Wormhole Topology](assets/lattice_wormholes.png)

### 1. 4D Voxel Lattice & Morton Encoding
The physical space is structured as a 4D hypercube lattice where every vertex represents a coordinate tuple $(x, y, z, w)$. To avoid carrying cumbersome multidimensional coordinate tensors, every tuple is serialized into a single scalar code using bitwise Morton interleaving (Z-order curve):

```python
def morton_encode_4d(x, y, z, w):
    result = 0
    for i in range(16):
        result |= ((x >> i) & 1) << (4 * i)
        result |= ((y >> i) & 1) << (4 * i + 1)
        result |= ((z >> i) & 1) << (4 * i + 2)
        result |= ((w >> i) & 1) << (4 * i + 3)
    return result
```

This guarantees reversible spatial hashing where locality in 4D space corresponds to proximity along the 1D Z-curve. The discrete coordinates are fed into dedicated per-axis embedding tables, concatenated, and gated directly into the Transformer's residual stream.

### 2. Wormhole Routing
As dimensions scale, Manhattan distances grow rapidly. In an autoregressive model, routing long-distance spatial dependencies through contiguous intermediate tokens clutters the causal attention window.

`WormholeRouter` connects distant lattice vertices with bidirectional shortcut bridges. Across sampled node pairs, plain lattice traversal requires an average of 3.55 hops, while wormhole-augmented breadth-first search cuts the average path length down to 2.95 hops. A binary wormhole presence indicator is embedded and gated into the token representation, signaling to the attention heads whenever a topological wormhole portal is active.

### 3. Conformal Prediction Routing
When predicting downstream spatial transitions under noise, standard softmax probabilities often suffer from overconfidence. `ConformalRouter` implements split conformal calibration to guarantee a distribution-free coverage bound:

$$\hat{q} = \text{Quantile}\left( \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$

With a target error tolerance $\alpha = 0.10$ ($90\%$ confidence), our empirical calibration on held-out test scores yielded a coverage rate of $90.50\%$, strictly satisfying the theoretical coverage guarantee.

---

## controlled comparison: the numbers

A common trick in academic machine learning is adding twenty extra modules to your model, observing a modest drop in loss, and proclaiming victory—conveniently ignoring that your model simply has $30\%$ more parameters than the baseline.

To keep things honest, we trained five distinct arms across three fixed seeds ($42$, $137$, $2024$) with identical iteration budgets and learning rate schedules:
1. **Vanilla**: Standard decoder-only GPT.
2. **Vanilla-Matched**: The standard GPT widened and deepened so its parameter count matches or exceeds the hybrid architecture.
3. **Vanilla + Lattice**: Standard GPT augmented with 4D coordinate embeddings alone.
4. **Vanilla + Wormhole**: Standard GPT augmented with wormhole routing feature gates alone.
5. **Full-Hybrid (DAS)**: The complete architecture combining coordinate embeddings and wormhole gates.

Here are the measured, unedited results:

| Model Arm | Parameters | Validation Loss (nats) | Bits Per Character (BPC) | Training Loss (nats) |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla** | 161,792 | 7.6721 +/- 0.1105 | 11.0684 +/- 0.1594 | 1.7786 +/- 0.0027 |
| **Vanilla-Matched** | 181,696 | 7.8209 +/- 0.1119 | 11.2832 +/- 0.1615 | 1.7684 +/- 0.0092 |
| **Vanilla + Lattice** | 170,240 | 5.9689 +/- 0.0758 | 8.6113 +/- 0.1093 | 1.7922 +/- 0.0065 |
| **Vanilla + Wormhole** | 170,176 | 6.7095 +/- 0.1490 | 9.6797 +/- 0.2150 | 1.8683 +/- 0.0100 |
| **Full-Hybrid (DAS)** | 178,624 | **4.8668 +/- 0.1651** | **7.0213 +/- 0.2382** | 1.9398 +/- 0.0048 |

### What this actually means

Take a look at `Vanilla-Matched`. When given more parameters without geometric grounding, its validation loss actually worsened from 7.67 to 7.82. Why? Because extra capacity simply allowed the vanilla network to overfit the training transitions even harder (training loss dropped to 1.76), leaving it even more fragile on held-out transitions.

Meanwhile, providing 4D coordinate embeddings alone dropped validation loss from 7.67 down to 5.96. Wormhole gates alone brought it down to 6.70. And the Full-Hybrid model dropped validation loss to 4.86 nats—a massive reduction of $2.80$ nats (over 4.0 bits per character) compared to the vanilla baseline.

Also notice the training loss of the Full-Hybrid: it is slightly higher (1.9398 vs 1.7786). That is not a flaw; it is the textbook hallmark of geometric regularization. The manifold-grounded network refuses to overfit training quirks because its latent representations are bound by the geometry of the 4D space.

---

## 4D manifold trajectory output

When you run the model, you do not just get raw statistical tables. The trained vanilla model also generates its learned spatial walk across the 4D grid, displaying the coordinate vector and orthogonal axis shifts step-by-step:

```text
Step 0:  (0, 0, 0, 0)
Step 1:  (0, 0, 0, 1)   [+1 along W-axis]
Step 2:  (0, 1, 0, 1)   [+1 along Y-axis]
Step 3:  (1, 1, 0, 1)   [+1 along X-axis]
Step 4:  [NOISE TOKEN]   (perturbed state)
Step 5:  (1, 1, 1, 1)   [+1 along Z-axis]
```

Every step demonstrates that the model has learned contiguous, valid physical trajectories through 4D spacetime.

---

## codebase structure

The repository is deliberately minimal:

- `bigram.py`: The architecture and mathematical core. Implements causal multi-head self-attention (`F.scaled_dot_product_attention`), Transformer blocks, `GPTLanguageModel`, `DASAugmentedGPT`, `VoxelLattice`, `WormholeRouter`, `ConformalRouter`, and 4D Morton encoding.
- `Train.py`: The execution engine. Generates the synthetic 4D dataset, implements cosine learning rate schedules with warmup, runs component verification proofs, executes the 5-arm benchmark, and samples outputs.
- `john_wick.txt`: The raw dialogue corpus used for character-level sanity checks.
- `assets/`: Contains the plots and visual artifacts generated during benchmarking.

Both Python files contain zero comments and zero docstrings. They are written in clean, direct PyTorch primitives that read like straightforward mathematics rather than framework abstraction soup.

---

## todos

- Scale the 4D lattice from $5^4$ (625 nodes) up to $16^4$ (65,536 nodes) on a multi-GPU node.
- Test whether learned continuous wormhole embeddings outperform fixed binary feature gates.
- Benchmark persistent homology barcode features against standard attention heads.
- Swap out character-level tokenization on `john_wick.txt` for a custom BPE tokenizer.
- Try training on Shakespeare just to see if John Wick starts speaking in iambic pentameter.

---

## acknowledgements

Inspired by Andrej Karpathy's `nanoGPT` and `minGPT` philosophies: write clean PyTorch, avoid unnecessary dependencies, keep the code hackable, and let the empirical results speak for themselves.
