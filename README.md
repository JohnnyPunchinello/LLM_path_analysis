# LLM Path Analysis — Mechanistic Interpretability via Path Distribution Theory

A mechanistic interpretability pipeline grounded in **Path Distribution Theory (PDT)**: transformers are residual networks where at every layer information either flows *through* a compute block (attention or MLP) or *bypasses* it via an identity skip connection. This project measures which paths are actually used for different tasks, and visualises the resulting active computation subgraphs.

---

## Quick start — Active Subgraph Visualiser

> **See what the model is doing with one command.**

```bash
# GPT-2, CPU, ~30 seconds
python active_subgraph_dot.py \
    --model gpt2 --device cpu \
    --suite reasoning \
    --out graphs/gpt2_reasoning

# Multiple models side-by-side
python active_subgraph_dot.py \
    --models gpt2 gpt2-medium EleutherAI/pythia-160m \
    --suite complexity_gradient --device cpu \
    --out graphs/compare

# Llama-3-8B on GPU (4-bit, ~5 GB VRAM)
python active_subgraph_dot.py \
    --model NousResearch/Meta-Llama-3-8B \
    --suite reasoning --device cuda \
    --out graphs/llama3
```

Each task produces three output files:
- `<stem>_taskN.png` — rendered graph (open directly)
- `<stem>_taskN.md`  — Mermaid.js source (paste at [mermaid.live](https://mermaid.live))
- `<stem>_taskN.dot` — Graphviz source (`dot -Tpng file.dot -o file.png`)

---

## Reading the graphs

```
         [Embedding]
              │  (teal = residual stream)
         ◇ r0 ◇ ──────────────────── residual stream checkpoint
        /α╲   \── H0 ──┐
       /   ╲   \── H1 ──┤
      /     ╲  ...      ├──> (Σ) ──── α ──> (Σ)
              \── H11 ──┘    attn-add      mlp-add
                               │               │
                            [FFN L0]          ...
              │
         ◇ r1 ◇  ← next checkpoint
```

| Element | Colour / style | Meaning |
|---|---|---|
| **◇ diamond node** | Teal fill | Residual stream checkpoint between layers |
| **Teal bold arrow** | `───α───>` | Active residual (skip) connection |
| **Red/orange ellipse** | Bold border | Active attention head; darker = higher attribution score |
| **Grey ellipse** | Faint border | Inactive attention head (below coverage threshold) |
| **Blue rectangle** | Bold border | Active FFN / MLP block |
| **Grey rectangle** | Faint border | Inactive FFN block |
| **(Σ)** | White circle | Residual addition node (attention add or MLP add) |

---

## Files

```
active_subgraph_dot.py   — Computational graph generator → PNG, SVG, Mermaid.js, DOT
active_subgraph_viz.py   — Quick matplotlib grid visualiser (no install required)
path_analyzer.py         — Core: DAG, Algorithm 1, AtP scoring, mass-coverage selection
experiment_runner.py     — Synergy-gap sweep across models × tasks  → CSV + plots
synergy_gap_experiment.py — Focused synergy-gap experiment with model-group presets
token_path_heatmap.py    — Per-token E[L] heatmap and time-series visualisation
skip_profile_experiment.py — WHERE does compute happen? (layer CoM, early/mid/late)
test_path_dp.py          — 48 unit tests for the DP path counter and mass-coverage rule
```

---

## Supported models

Any model loadable by [TransformerLens](https://github.com/neelnanda-io/TransformerLens) works. Tested models:

| Model | Parameters | Architecture | VRAM (4-bit) | Notes |
|---|---|---|---|---|
| `gpt2` | 117 M | Sequential | CPU only | Fastest, good for development |
| `gpt2-medium` | 345 M | Sequential | CPU only | |
| `gpt2-large` | 774 M | Sequential | CPU only | |
| `gpt2-xl` | 1.5 B | Sequential | ~1.5 GB | |
| `EleutherAI/pythia-70m` | 70 M | Sequential | CPU only | Tiny, instant |
| `EleutherAI/pythia-160m` | 160 M | Sequential | CPU only | |
| `EleutherAI/pythia-410m` | 410 M | Sequential | CPU only | |
| `EleutherAI/pythia-1b` | 1 B | Sequential | ~1 GB | |
| `EleutherAI/pythia-2.8b` | 2.8 B | Sequential | ~2 GB | |
| `EleutherAI/pythia-6.9b` | 6.9 B | Sequential | ~4 GB | |
| `EleutherAI/pythia-12b` | 12 B | Sequential | ~7 GB | |
| `EleutherAI/gpt-j-6b` | 6 B | **Parallel** | ~4 GB | Parallel attn+MLP per layer |
| `NousResearch/Meta-Llama-3-8B` | 8 B | Sequential (RoPE+RMSNorm) | ~5 GB | ✅ Non-gated |
| `meta-llama/Meta-Llama-3-8B` | 8 B | Sequential (RoPE+RMSNorm) | ~5 GB | Requires Meta approval |
| `meta-llama/Meta-Llama-3-70B` | 70 B | Sequential (RoPE+RMSNorm) | ~38 GB | Requires Meta approval |

> **Llama models** require `fold_ln=False, center_writing_weights=False, center_unembed=False` — these are set automatically.

---

## Task suites (`--suite`)

### `quick` — 3 tasks, fast smoke-test
| # | Prompt | Label |
|---|---|---|
| 1 | `The cat sat on the mat.` | Simple sentence |
| 2 | `What is 7 times 8? The answer is` | Arithmetic |
| 3 | `Alice is the mother of Bob. Bob is the mother of Carol. Who is Alice's grandchild? The answer is` | 2-hop reasoning |

### `complexity_gradient` — 8 tasks, full difficulty ramp
| # | Prompt | Label | Capability |
|---|---|---|---|
| 1 | `The dog barked loudly.` | Lexical | Surface form |
| 2 | `The keys to the cabinet are on the table. The keys` | Subject-verb agreement | Syntax (attractor) |
| 3 | `The capital of France is` | 1-hop factual | World knowledge |
| 4 | `17 plus 28 equals` | Arithmetic | Numeric computation |
| 5 | `Alice is the mother of Bob. Bob is the mother of Carol. Alice's grandchild is` | 2-hop reasoning | Compositional |
| 6 | `Alice is the parent of Bob. Bob is the parent of Carol. Carol is the parent of Dana. Alice's great-grandchild is` | 3-hop reasoning | Deep compositional |
| 7 | `All mammals breathe air. Dolphins are mammals. Therefore, dolphins` | Logical syllogism | Deductive reasoning |
| 8 | `Paris is to France as Berlin is to` | Analogy | Relational reasoning |

### `syntax` — 5 tasks, subject-verb and long-range agreement
| # | Prompt | Label |
|---|---|---|
| 1 | `The cat sat on the mat.` | Simple SVO |
| 2 | `The keys to the cabinet are on the table. The keys` | PP attractor |
| 3 | `The man who the dogs chased ran. The man` | Object-extracted RC |
| 4 | `She said that he believed that they would come. They` | Embedded clause |
| 5 | `Either the manager or the employees are responsible. They` | Either-or agreement |

### `arithmetic` — 6 tasks, trivial addition to word problems
| # | Prompt | Label |
|---|---|---|
| 1 | `2 + 2 =` | Trivial addition |
| 2 | `17 + 28 =` | 2-digit addition |
| 3 | `7 times 8 equals` | Single-digit multiplication |
| 4 | `144 divided by 12 equals` | Division |
| 5 | `What is 15% of 200? The answer is` | Percentage |
| 6 | `If a train travels at 60 mph for 2.5 hours, it covers` | Word problem |

### `reasoning` — 6 tasks, 1-hop to counterfactual
| # | Prompt | Label | Depth |
|---|---|---|---|
| 1 | `Alice is the mother of Bob. Alice's child is` | 1-hop chain | k=1 |
| 2 | `Alice is the mother of Bob. Bob is the mother of Carol. Alice's grandchild is` | 2-hop chain | k=2 |
| 3 | `Alice → Bob → Carol → Dana. Alice's great-grandchild is` | 3-hop chain | k=3 |
| 4 | `All birds have wings. A penguin is a bird. Therefore, a penguin has` | Categorical syllogism | — |
| 5 | `No reptiles are warm-blooded. All mammals are warm-blooded. Therefore, snakes are` | Negation + deduction | — |
| 6 | `In a world where cats bark and dogs meow, if you hear barking outside you think it is a` | Counterfactual | — |

### `world_knowledge` — 6 tasks, factual recall
| # | Prompt | Label |
|---|---|---|
| 1 | `The capital of Japan is` | Capital city |
| 2 | `Shakespeare wrote the play Hamlet. The author of Hamlet is` | Author recall |
| 3 | `Water is made of hydrogen and` | Chemical composition |
| 4 | `The theory of relativity was developed by` | Scientific attribution |
| 5 | `In 1969, Neil Armstrong became the first person to walk on the` | Historical event |
| 6 | `The largest planet in the solar system is` | Astronomy fact |

---

## CLI reference — `active_subgraph_dot.py`

```bash
python active_subgraph_dot.py [options]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt2` | Single model name (used when `--models` is not set) |
| `--models` | — | Space-separated list of model names for cross-architecture comparison |
| `--suite` | `quick` | Named task suite (see above) |
| `--tasks` | — | Custom prompts, overrides `--suite` |
| `--labels` | auto | Short label per task |
| `--mass_coverage` | `0.9` | Nucleus coverage target: the minimum edges covering 90% of attribution mass are kept active |
| `--head_threshold` | `0.15` | Head active if score ≥ threshold × max-score in its layer |
| `--device` | `cuda` | Falls back to `cpu` automatically |
| `--hf_token` | — | HuggingFace token for gated repos (or set `HF_TOKEN` env var) |
| `--out` | `graphs/subgraph` | Output path stem; model name appended when `--models` is used |

---

## CLI reference — `active_subgraph_viz.py` (matplotlib grid)

Simpler alternative: produces a single PNG grid without needing Graphviz.

```bash
python active_subgraph_viz.py \
    --model gpt2 --device cpu \
    --tasks "The cat sat on the mat." "7 times 8 equals" \
    --task_labels "Simple" "Arithmetic" \
    --mass_coverage 0.9 --out grid.png
```

---

## Installation

```bash
pip install transformer_lens datasets networkx matplotlib torch einops

# For PNG/SVG rendering from active_subgraph_dot.py:
pip install graphviz
brew install graphviz          # macOS
# sudo apt install graphviz   # Linux

# For 4-bit quantisation (Llama-3, large models):
pip install bitsandbytes accelerate
```

---

## HuggingFace authentication (gated models)

`NousResearch/Meta-Llama-3-8B` is a non-gated community mirror with identical weights and **requires no token**. For the official `meta-llama/*` repos (requires Meta approval at huggingface.co/meta-llama):

```bash
# Option 1: CLI flag
python active_subgraph_dot.py --model meta-llama/Meta-Llama-3-8B --hf_token hf_...

# Option 2: environment variable (recommended)
export HF_TOKEN=hf_...
python active_subgraph_dot.py --model meta-llama/Meta-Llama-3-8B

# Option 3: cached login
huggingface-cli login
```

---

## Methodology

### DAG construction
The transformer is mapped to a block-level Directed Acyclic Graph.

**Sequential (Pre-LN/RMS — GPT-2, Pythia, Llama):**
```
resid_pre_l  ──[skip Δ=0]──▶  resid_mid_l  ──[skip Δ=0]──▶  resid_post_l
             ──[attn Δ=1]──▶               ──[mlp  Δ=1]──▶
```

**Parallel (GPT-J — attn and MLP in parallel):**
```
resid_pre_l  ──[skip Δ=0]──▶
             ──[attn Δ=1]──▶  resid_post_l
             ──[mlp  Δ=1]──▶
```

### Mass-coverage active edge selection (nucleus rule)
> Select the **fewest edges** whose combined attribution mass covers ≥ 90% of total attribution.

```
sorted_desc = sort(all_edge_scores, descending=True)
cumsum      = cumulative_sum(sorted_desc)
i           = smallest index where cumsum[i] >= 0.90 × total
ε           = sorted_desc[i]
active      = {edge : score_edge >= ε}
```

Simple tokens → concentrated attribution → small, sparse active subgraph.  
Complex tokens → spread attribution → large, dense active subgraph.

### Attribution Patching (AtP) — per-head scores
Edge score = gradient × activation at `hook_z` (head output before W_O):

```
s(l, h) = mean_{seq, d_head} | ∂logit/∂z[l,h] · z[l,h] |
```

Anchored at `blocks.0.hook_resid_pre` (detached float32 leaf) to avoid propagating through quantised embeddings.

### Path entropy and synergy gap
| Metric | Formula |
|---|---|
| Analytical entropy | H(π) = −Σ_l π(l) log₂ π(l), all edges active |
| Empirical entropy | H(π̂) = same, restricted to active subgraph |
| Synergy Gap | ΔH = H(π) − H(π̂) |
| Mean path length | E[L] = Σ_l l · π̂(l) |
| Tail-mass ratio | τ_k = P(L > k) / P(L ≤ k) |

---

## Dependencies

| Package | Purpose |
|---|---|
| [TransformerLens](https://github.com/neelnanda-io/TransformerLens) | Hook-based transformer inference |
| [PyTorch](https://pytorch.org/) ≥ 2.0 | Model execution |
| [HuggingFace Transformers](https://huggingface.co/docs/transformers) | Model loading |
| [NetworkX](https://networkx.org/) | DAG construction |
| [NumPy](https://numpy.org/) / [Matplotlib](https://matplotlib.org/) | Computation / plotting |
| [graphviz](https://pypi.org/project/graphviz/) + `dot` binary | PNG / SVG rendering |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) *(optional)* | 4-bit / 8-bit quantisation |
| [accelerate](https://github.com/huggingface/accelerate) *(optional)* | Multi-GPU device mapping |

---

## License

MIT
