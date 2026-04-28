# LLM Path Analysis — Mechanistic Interpretability via Path Distribution Theory

A mechanistic interpretability pipeline that tracks **Path Length Distributions**, **Empirical Path Entropy**, and **Skip Profiles** across LLM architectures and tasks, grounded in Path Distribution Theory (PDT).

## Overview

Modern transformers are residual networks: at every layer, information can either flow through a computation block (attention or MLP) or bypass it via the identity skip connection. This creates an exponential number of possible "paths" through the network. This project measures two complementary things:

**Global path statistics (per prompt):**
- **Analytical Path Entropy H(π)** — entropy of the path-length distribution implied by architecture alone (architecture constant)
- **Empirical Path Entropy H(π̂)** — entropy when only *task-relevant* paths are counted (via Attribution Patching)
- **Synergy Gap** — H(π) − H(π̂): how much the task prunes the theoretical path space
- **Mean Path Length E[L]** and **Tail-Mass Ratio τ_k**

**Per-layer skip profiles (new):**
- **Layer compute weight w_t(ℓ)** — what fraction of total attribution mass is concentrated at layer ℓ
- **Center of Mass (CoM)** — centroid of compute across the layer stack
- **Early / Mid / Late compute density (ES, MS, LS)** — compute concentration in each third
- **Causal ablation** — accuracy drop after zeroing each layer region (H3: double dissociation)

## Architecture Support

| Architecture | Branch Type | Example Models | Max Path Len |
|---|---|---|---|
| Sequential Pre-LN/RMS | Binary × 2 per layer | Llama-3-8B, GPT-2, Pythia | 2 × n\_layers |
| Parallel (GPT-J style) | Ternary per layer | EleutherAI/gpt-j-6b | n\_layers |
| Attention-only | Binary per layer | custom | n\_layers |

## Files

```
path_analyzer.py              — Core PathAnalyzer class (DAG, Algorithm 1, AtP, entropy)
                                + select_active_edges_by_mass_coverage() (nucleus rule)
active_subgraph_viz.py        — Active subgraph visualiser: see WHICH blocks fire per task
experiment_runner.py          — Synergy-gap sweep across models × tasks; CSV + plots
synergy_gap_experiment.py     — Focused synergy-gap experiment with model-group presets
token_path_heatmap.py         — Per-token E[L] heatmap and time-series visualization
skip_profile_experiment.py    — Skip profile analysis: WHERE does compute happen?
test_path_dp.py               — 48 unit tests for the DP path counter and mass-coverage rule
```

## Installation

```bash
pip install transformer_lens datasets networkx matplotlib torch

# For 4-bit / 8-bit quantisation (A100 / RTX 4090 recommended for Llama-3-8B):
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

# Empirical (data-driven via Attribution Patching + mass-coverage selection)
tokens = model.to_tokens("The quick brown fox jumps", prepend_bos=True)
emp, info = analyzer.empirical_path_distribution(tokens)
print(f"Empirical  H = {emp.entropy:.3f} bits  active_attn = {info['n_active_attn']}")
print(f"Synergy Gap  = {ana.entropy - emp.entropy:.3f} bits")
```

---

## Quickest Start — Active Subgraph Visualiser

> **If you want to see what the network is doing with one command, start here.**

`active_subgraph_viz.py` draws the active computation subgraph for a list of task prompts side-by-side. No CSV, no aggregate statistics — just a picture of which blocks fire and how they chain together.

### What you see

```
     [A0] [A1] [A2]  ...  [A_{L-1}]      ← attention blocks (above backbone)
  ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●   ← residual stream backbone
     [M0] [M1] [M2]  ...  [M_{L-1}]      ← MLP blocks (below backbone)
```

| Visual element | Meaning |
|---|---|
| **Orange/red box** (bold border) | Active attention block; colour intensity = attribution score |
| **Blue box** (bold border) | Active MLP block; colour intensity = attribution score |
| **Grey box** (faint border) | Inactive block (below mass-coverage threshold) |
| **Bold black backbone** | Active residual (skip) connection — both flanking stream nodes active |
| **Dashed grey backbone** | Inactive residual connection |
| **Diagonal lines** through a box | Active compute arc: stream → block → stream |

Active blocks are selected by the **nucleus (mass-coverage) rule**: the minimum number of edges whose combined attribution mass covers ≥ 90% of the total. Simpler tokens recruit fewer blocks; complex reasoning tokens recruit more.

### Usage

```bash
# GPT-2 on CPU — runs in ~30 seconds
python active_subgraph_viz.py \
    --model gpt2 \
    --device cpu \
    --tasks "The cat sat on the mat." \
            "What is 7 times 8? The answer is" \
            "Alice is the mother of Bob. Bob is the mother of Carol. Who is Alice's grandchild?" \
    --task_labels "Simple sentence" "Arithmetic" "2-hop reasoning" \
    --mass_coverage 0.9 \
    --out subgraphs.png

# Llama-3-8B on GPU with 4-bit quantisation (~5 GB VRAM)
python active_subgraph_viz.py \
    --model NousResearch/Meta-Llama-3-8B \
    --device cuda \
    --tasks "The cat sat on the mat." \
            "Prove that the square root of 2 is irrational." \
    --task_labels "Simple" "Hard math" \
    --mass_coverage 0.9 \
    --out llama_subgraphs.png

# Sparse view — fewer active blocks, structural differences become obvious
python active_subgraph_viz.py --model gpt2 --device cpu --mass_coverage 0.5
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt2` | Any TransformerLens-compatible model name |
| `--tasks` | 4 built-in examples | Space-separated list of prompt strings |
| `--task_labels` | `Task 1, 2, …` | Short panel label per task |
| `--mass_coverage` | `0.9` | Nucleus coverage target (lower → sparser graph) |
| `--device` | `cuda` | Falls back to `cpu` automatically |
| `--hf_token` | `None` | HuggingFace token for gated repos |
| `--out` | `active_subgraphs.png` | Output path (`.png`, `.pdf`, `.svg`) |
| `--dpi` | `150` | Output resolution |

---

## Experiment Scripts

### 1. `skip_profile_experiment.py` — Skip Profile Analysis (recommended starting point)

Answers the question: **Where along the layer stack does each capability class concentrate its compute?**

**Core quantity — layer compute weight:**

```
w_t(ℓ) = (attn_score_t[ℓ] + mlp_score_t[ℓ]) / Σ_ℓ (attn_score_t[ℓ] + mlp_score_t[ℓ])
```

High `w_t(ℓ)` → compute-edge dominant at layer ℓ (low skip probability).
Low `w_t(ℓ)` → skip-edge dominant at layer ℓ (high skip probability).

**Six built-in test conditions (no downloads required):**

| # | Condition | Capability | Predicted layer region |
|---|---|---|---|
| 1 | Subject-Verb Agreement | Syntax (attractor interference) | Early [0, L/3) |
| 2 | Factual Recall | World-knowledge retrieval | Late [2L/3, L) |
| 3a | 1-hop Bridge | Direct entity recall | Intermediate |
| 3b | 2-hop Bridge | Compositional reasoning | Middle [L/3, 2L/3) |
| 4a | Induction Copy | Literal n-gram repetition | Early–mid |
| 4b | Compositional Copy | Entity-attribute retrieval | Late |

**Three falsifiable hypotheses:**
- **H1** (Early/Late dissociation): CoM(SV) < CoM(2-hop) < CoM(Factual)
- **H2** (Mid-layer bottleneck): MS(2-hop) > MS(1-hop) and MS(2-hop) > MS(Factual)
- **H3** (Causal double dissociation): zeroing each layer region collapses only that region's capability

**Usage:**

```bash
# Quick smoke test (GPT-2, CPU, 5 examples per condition)
python skip_profile_experiment.py --model gpt2 --device cpu --n_samples 5

# Full run (Llama-3-8B, 4-bit NF4, GPU)
python skip_profile_experiment.py \
    --model NousResearch/Meta-Llama-3-8B \
    --quant 4bit --device cuda

# GPT-J parallel-architecture comparison
python skip_profile_experiment.py \
    --model EleutherAI/gpt-j-6b --quant 8bit --device cuda --n_samples 20

# Profile only (no ablation, much faster)
python skip_profile_experiment.py --model gpt2 --device cpu --no_ablation

# Subset of conditions
python skip_profile_experiment.py \
    --model gpt2 --device cpu \
    --conditions "SV-agreement,2-hop bridge,Factual-recall"
```

**Outputs** (in `results/skip_profile/`):

| File | Description |
|---|---|
| `skip_profile_heatmap.png` | 6-row × L-col heatmap of mean w̄_t(ℓ), rows ordered by CoM |
| `skip_profile_com.png` | Horizontal bar chart: CoM per condition with ±1 SD error bars |
| `ablation_results.png` | Grouped bar: accuracy drop after zeroing each layer region |
| `skip_profile_summary.csv` | Per-condition CoM, ES, MS, LS mean ± std |
| `ablation_results.csv` | Accuracy before / after ablation, per condition × region |

**CLI Options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt2` | HuggingFace model ID |
| `--device` | `cuda` if available, else `cpu` | Compute device |
| `--quant` | `none` | `4bit` (NF4) \| `8bit` (int8) \| `none` |
| `--hf_token` | None | Token for gated repos (`meta-llama/*`) |
| `--n_samples` | `20` | Examples per condition (max 20) |
| `--output_dir` | `results/skip_profile` | Output directory |
| `--no_ablation` | False | Skip causal ablation (profile only) |
| `--conditions` | all 6 | Comma-separated subset of condition names |

---

### 2. `experiment_runner.py` — Synergy Gap Sweep

Measures path entropy and synergy gap across models × NLP tasks.

```bash
# Quick test (GPT-2, CPU, 5 samples)
python experiment_runner.py --models gpt2 --n_samples 5 --device cpu

# Pythia scaling series
python experiment_runner.py --model_group pythia \
    --tasks sst2,boolq,arc_easy,hellaswag

# Llama-3-8B (4-bit, GPU)
python experiment_runner.py --model_group llama \
    --tasks sst2,boolq,arc_easy,hellaswag,gsm8k \
    --quant 4bit

# Full cross-architecture sweep
python experiment_runner.py --model_group all --n_samples 50
```

**Model groups:** `pythia`, `gpt2`, `neo`, `llama`, `llama_gated`, `llama_instruct`, `large`, `all`

---

### 3. `synergy_gap_experiment.py` — Focused Synergy Gap Analysis

```bash
# Small models (CPU smoke test)
python synergy_gap_experiment.py --model_group small \
    --tasks sst2,boolq,hellaswag --n_samples 5 --device cpu

# Llama-3-8B (4-bit)
python synergy_gap_experiment.py --model_group llama \
    --tasks sst2,piqa,boolq,arc_easy,copa,hellaswag,arc_challenge,lambada \
    --n_samples 50

# Resume after interruption
python synergy_gap_experiment.py --model_group pythia \
    --tasks sst2,boolq --n_samples 50 \
    --resume results/synergy_gap_results.csv

# Regenerate figures from saved CSV (no model loading)
python synergy_gap_experiment.py --plot_only results/synergy_gap_results.csv
```

---

### 4. `token_path_heatmap.py` — Per-Token Path Heatmap

Visualizes E[L] (mean path length) per token position for a given prompt.

```bash
# Single prompt
python token_path_heatmap.py \
    --model gpt2 --device cpu \
    --prompt "The keys to the cabinet are on the table"

# Save JSON
python token_path_heatmap.py \
    --model NousResearch/Meta-Llama-3-8B --quant 4bit --device cuda \
    --prompt "Who wrote Hamlet?" \
    --output_dir results/heatmap
```

---

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

For a full sequential L-layer model (all edges active) this yields Binomial(2L, 0.5) — E[L] = L, total paths = 4^L.

### Mass-Coverage Active Edge Selection (nucleus rule)

Instead of a fixed percentage threshold, the active subgraph is selected by a **minimum-set nucleus rule**:

> Select the fewest edges whose combined attribution mass covers ≥ 90% of the total attribution mass.

```python
# Pseudocode
sorted_desc = sort(all_scores, descending=True)
cumsum      = cumulative_sum(sorted_desc)
i           = smallest index where cumsum[i] >= 0.90 × total
epsilon     = sorted_desc[i]
active      = {edge : score_edge >= epsilon}
```

Simple tokens → attribution concentrated in few layers → small active set.
Complex tokens → attribution spread → large active set.

### Attribution Patching (AtP)

Edge scores use a first-order linear approximation:

```
AtP(e) = ∇_e [logit]_baseline · (a_e^clean − a_e^baseline)
```

With a zero baseline this reduces to gradient × activation. The gradient is anchored at `blocks.0.hook_resid_pre` (detached float32 leaf) and backpropagated from the target token's logit.

### Skip Profile Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Layer compute weight | w_t(ℓ) = c_t(ℓ) / Σ_ℓ c_t(ℓ) | Fraction of compute at layer ℓ |
| Center of Mass | CoM_t = Σ_ℓ ℓ · w_t(ℓ) | Centroid of compute layers |
| Early density | ES_t = mean w_t(ℓ), ℓ ∈ [0, L/3) | Compute in early third |
| Mid density | MS_t = mean w_t(ℓ), ℓ ∈ [L/3, 2L/3) | Compute in middle third |
| Late density | LS_t = mean w_t(ℓ), ℓ ∈ [2L/3, L) | Compute in late third |

### Global Path Distribution Metrics

| Metric | Formula |
|---|---|
| Empirical Path Entropy | H(π̂) = −Σ_l π̂(l) log₂ π̂(l) |
| Mean Path Length | E[L] = Σ_l l · π̂(l) |
| Tail-Mass Ratio | τ_k = P(L > k) / P(L ≤ k), k = ⌊E[L]⌋ |
| Synergy Gap | H(π) − H(π̂) |

---

## HuggingFace Authentication

Non-gated models (GPT-2, Pythia, GPT-J, **NousResearch/Meta-Llama-3-8B**) work without a token.

For gated `meta-llama/*` models (requires Meta approval at huggingface.co/meta-llama):

```bash
# Option 1: CLI flag
python skip_profile_experiment.py --model meta-llama/Meta-Llama-3-8B --hf_token YOUR_TOKEN

# Option 2: environment variable
export HF_TOKEN=your_token
python skip_profile_experiment.py --model meta-llama/Meta-Llama-3-8B

# Option 3: cached login
huggingface-cli login
python skip_profile_experiment.py --model meta-llama/Meta-Llama-3-8B
```

---

## CSV Schema — Skip Profile Summary

| Column | Description |
|---|---|
| `condition` | Test condition name |
| `n_samples` | Number of examples processed |
| `n_layers` | Number of transformer layers |
| `early_end` | Last layer index of early third |
| `mid_end` | Last layer index of middle third |
| `com_mean` | Mean center of mass across examples |
| `com_std` | Std dev of CoM |
| `es_mean` | Mean early-third compute density |
| `ms_mean` | Mean middle-third compute density |
| `ls_mean` | Mean late-third compute density |
| `accuracy` | Baseline task accuracy |

## CSV Schema — Synergy Gap / Path Metrics

| Column | Description |
|---|---|
| `model` | HuggingFace model ID |
| `task` | Task name |
| `task_complexity` | 0 / 1 / 2 (Low / Medium / High) |
| `accuracy` | Greedy next-token accuracy |
| `analytical_entropy` | H(π) — architecture constant |
| `empirical_entropy` | H(π̂) — mean over samples |
| `empirical_entropy_std` | Std of H(π̂) over samples |
| `synergy_gap` | analytical\_entropy − empirical\_entropy |
| `empirical_mean_path` | Mean E[L] over samples |
| `empirical_tail_ratio` | Mean τ_k over samples |
| `n_layers` | Number of transformer layers |
| `architecture` | sequential / parallel / attn-only |
| `mass_coverage` | AtP nucleus coverage target (default 0.90) |
| `mean_epsilon` | Mean per-sample AtP threshold |

---

## Dependencies

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — hook-based transformer inference
- [PyTorch](https://pytorch.org/) ≥ 2.0
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — SST-2, BoolQ, GSM8K (for synergy-gap experiments)
- [NetworkX](https://networkx.org/) — DAG construction and validation
- [NumPy](https://numpy.org/) / [Matplotlib](https://matplotlib.org/)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) *(optional)* — 4-bit / 8-bit quantisation
- [accelerate](https://github.com/huggingface/accelerate) *(optional)* — multi-GPU device map

## License

MIT
