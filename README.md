# LLM Path Analysis — Mechanistic Interpretability via Path Distribution Theory

A mechanistic interpretability pipeline that tracks **Path Length Distributions** and **Empirical Path Entropy** across LLM architectures and tasks, grounded in Path Distribution Theory.

## Overview

Modern transformers are residual networks: at every layer, information can either flow through a computation block (attention or MLP) or bypass it via the identity skip connection. This creates an exponential number of possible "paths" through the network. This project measures:

- **Analytical Path Entropy H(π)** — the entropy of the path-length distribution implied by the model's architecture alone (a constant per model)
- **Empirical Path Entropy H(π̂)** — the entropy when only *task-relevant* paths are counted (determined via Attribution Patching)
- **Synergy Gap** — H(π) − H(π̂): how much the task prunes the theoretical path space
- **Mean Path Length E[L]** and **Tail-Mass Ratio τ_k**

## Architecture Support

| Architecture | Branch Type | Example Models | Max Path Len |
|---|---|---|---|
| Sequential Pre-LN/RMS | Binary × 2 per layer | Llama-3-8B, GPT-2 | 2 × n_layers |
| Parallel (GPT-J style) | Ternary per layer | EleutherAI/gpt-j-6b | n_layers |
| Attention-only | Binary per layer | custom | n_layers |

## Files

```
path_analyzer.py       — Core PathAnalyzer class (DAG, Algorithm 1, AtP, entropy)
experiment_runner.py   — Experiment loop, CSV export, matplotlib visualisation
```

## Installation

```bash
pip install transformer_lens datasets networkx matplotlib torch
# For 8-bit quantisation on A100:
pip install bitsandbytes accelerate
```

## Quick Start

```python
from transformer_lens import HookedTransformer
from path_analyzer import PathAnalyzer
import torch

model = HookedTransformer.from_pretrained("gpt2", dtype=torch.float16)
analyzer = PathAnalyzer(model)

# Architecture-level (no data needed)
ana = analyzer.analytical_path_distribution()
print(f"Analytical H = {ana.entropy:.3f} bits  E[L] = {ana.mean_path_length:.2f}")

# Empirical (data-driven via Attribution Patching)
tokens = model.to_tokens("The quick brown fox jumps", prepend_bos=True)
emp, info = analyzer.empirical_path_distribution(tokens, epsilon_quantile=0.25)
print(f"Empirical  H = {emp.entropy:.3f} bits  active_attn = {info['n_active_attn']}")
print(f"Synergy Gap  = {ana.entropy - emp.entropy:.3f} bits")
```

## Experiment Runner

```bash
# Quick test (GPT-2, CPU, 5 samples)
python experiment_runner.py --models gpt2 --n_samples 5 --device cpu

# Full A100 run across all three models and tasks
python experiment_runner.py \
    --models "meta-llama/Llama-3-8B,EleutherAI/gpt-j-6b,tiiuae/falcon-7b" \
    --tasks  sst2,boolq,gsm8k \
    --n_samples 50
```
```bash
# Quick smoke test (CPU, 4 small models, 3 tasks)
python synergy_gap_experiment.py --model_group small \
    --tasks sst2,boolq,hellaswag --n_samples 5 --device cpu

# Sequential vs parallel architecture contrast
python synergy_gap_experiment.py --model_group neo \
    --tasks sst2,piqa,boolq,arc_easy,copa,hellaswag,arc_challenge,lambada \
    --n_samples 50

# OPT (post-LN) vs Pythia (pre-RMSNorm) normalization contrast
python synergy_gap_experiment.py \
    --models "facebook/opt-125m,facebook/opt-1.3b,EleutherAI/pythia-160m,EleutherAI/pythia-1b" \
    --tasks sst2,cola,boolq,arc_easy,openbookqa,hellaswag,lambada --n_samples 50

# Resume after interruption
python synergy_gap_experiment.py --model_group pythia \
    --tasks sst2,piqa,cola,boolq,arc_easy,copa,hellaswag,lambada \
    --n_samples 50 --resume results/synergy_gap_results.csv

# Regenerate figures from existing CSV (no model loading)
python synergy_gap_experiment.py --plot_only results/synergy_gap_results.csv
```

Outputs are written to `results/`:
- `path_metrics.csv` — full results table
- `entropy_vs_complexity.png` — 3-panel figure (see below)

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--models` | Llama-3-8B, GPT-J-6B, Falcon-7B | Comma-separated HuggingFace model IDs |
| `--tasks` | sst2,boolq,gsm8k | Task names (Low/Medium/High complexity) |
| `--n_samples` | 50 | Samples per task |
| `--device` | cuda | Compute device |
| `--epsilon_quantile` | 0.25 | AtP threshold quantile (0.25 → top-75% active) |
| `--output_dir` | results/ | Output directory |

## Methodology

### DAG Construction

The transformer is mapped to a block-level Directed Acyclic Graph (DAG).

**Sequential (Pre-LN/RMS — Llama, GPT-2):**
```
resid_pre_l  ──[skip, Δ=0]──▶  resid_mid_l  ──[skip, Δ=0]──▶  resid_post_l
             ──[attn, Δ=1]──▶               ──[mlp,  Δ=1]──▶
```

**Parallel (GPT-J):**
```
resid_pre_l  ──[skip, Δ=0]──▶
             ──[attn, Δ=1]──▶  resid_post_l
             ──[mlp,  Δ=1]──▶
```

### Algorithm 1 — DP Path Counter

Dynamic programming over the DAG. At each binary branch point:

```
counts_new[l] = counts[l]      # skip  (always active)
              + counts[l-1]    # compute (if edge ∈ G_active)
```

At each ternary branch point (parallel):

```
counts_new[l] = counts[l]                      # skip
              + has_attn × counts[l-1]          # attn compute
              + has_mlp  × counts[l-1]          # mlp  compute
```

For a full sequential L-layer model (all edges active) this yields Binomial(2L, l) — E[L] = L, total paths = 4^L.

### Attribution Patching (AtP)

Edge scores use a first-order linear approximation of activation patching:

```
AtP(e) = ∇_e [logit]_baseline · (a_e^clean − a_e^baseline)
```

With a zero baseline this reduces to gradient × activation. The gradient is computed by anchoring the computation graph at `blocks.0.hook_resid_pre` (detached, `requires_grad=True`) and backpropagating from the target token's logit.

### Active Subgraph G_active

Edges with AtP score > ε are included in G_active. ε is set per-task as the `epsilon_quantile`-th quantile of all attribution scores across samples (default: 25th percentile → top 75% of edges active).

### Metrics

| Metric | Formula |
|---|---|
| Empirical Path Entropy | H(π̂) = −Σ_l π̂(l) log₂ π̂(l) |
| Mean Path Length | E[L] = Σ_l l · π̂(l) |
| Tail-Mass Ratio | τ_k = P(L > k) / P(L ≤ k), k = ⌊E[L]⌋ |
| Synergy Gap | H(π) − H(π̂) |

## Output Figure

Three panels are produced comparing models across task complexity (SST-2 < BoolQ < GSM8K):

1. **Empirical Path Entropy H(π̂)** with ±1 std error bars
2. **Synergy Gap** — how much the task narrows the path distribution
3. **Mean Path Length E[L]** vs analytical reference (dotted)

## CSV Schema

| Column | Description |
|---|---|
| `model` | HuggingFace model ID |
| `task` | sst2 / boolq / gsm8k |
| `task_complexity` | 0 / 1 / 2 |
| `accuracy` | Greedy next-token accuracy |
| `analytical_entropy` | H(π) — architecture constant |
| `empirical_entropy` | H(π̂) — mean over samples |
| `empirical_entropy_std` | Std of H(π̂) over samples |
| `synergy_gap` | analytical_entropy − empirical_entropy |
| `empirical_mean_path` | Mean E[L] over samples |
| `empirical_tail_ratio` | Mean τ_k over samples |
| `n_layers` | Number of transformer layers |
| `architecture` | sequential / parallel / attn-only |
| `epsilon` | Task-level AtP threshold used |

## Dependencies

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — hook-based transformer inference
- [PyTorch](https://pytorch.org/) ≥ 2.0
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — SST-2, BoolQ, GSM8K
- [NetworkX](https://networkx.org/) — DAG construction and validation
- [NumPy](https://numpy.org/) / [Matplotlib](https://matplotlib.org/)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) *(optional)* — 8-bit quantisation

## License

MIT
