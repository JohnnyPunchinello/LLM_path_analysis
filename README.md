# LLM Path Analysis — Dead-Head Recruitment & the Active Subgraph

This repository measures **how a transformer allocates its attention machinery to a task**.
For a given prompt we ask, layer by layer and head by head, *which units are actually doing
work and which are idle ("dead")*, and how that pattern changes as the task's demands change.

The central object is the **active subgraph**: the set of attention heads and FFN blocks that
carry non-negligible signal for a prediction. Its complement — the **dead-head landscape**
`d(ℓ)` = number of dead heads at layer `ℓ` — turns out to be a graded, readable fingerprint of
task complexity.

**Headline hypothesis (two axes of complexity):**

| Axis | Task property | Network response | Status |
|---|---|---|---|
| **Vertical (recruitment)** | more *parallel width* `W` (independent sub-computations) | recruits more **attention heads per layer** | ✅ confirmed |
| **Horizontal (frontier)** | more *serial depth* `D` (sequential composition steps) | recruits more **layers** (active frontier extends) | ⚠️ confirmed on GPT-2/Pythia + 8B; null on 70B (see Results) |

---

## Contents

- [Install & setup](#install--setup)
- [The three scripts](#the-three-scripts)
- [How things are computed](#how-things-are-computed) — dead-head definition, the three metrics
- [Task suites (with exact prompts)](#task-suites-with-exact-prompts)
- [Current results](#current-results)
- [Outputs & data folders](#outputs--data-folders)
- [The paper](#the-paper)
- [Caveats](#caveats)

---

## Install & setup

Designed to run in Google Colab on an A100 (80 GB) for the large models, or locally for the
small ones.

```bash
git clone https://github.com/JohnnyPunchinello/LLM_path_analysis.git
pip install transformer_lens datasets networkx matplotlib torch
pip install -qU bitsandbytes accelerate      # for large-model loading / offload
```

Gated models (Llama-3) need a Hugging Face token with access approval; pass it via
`--hf_token`. **Never commit your token** — pass it at runtime only.

Model loading is automatic and size-aware (`load_model` in `active_subgraph_dot.py`):
- **Small models** (GPT-2, Pythia) load via TransformerLens.
- **Large models** (Llama-3-8B/70B) load as a raw Hugging Face model wrapped in
  `_HFHookedModel`, which attaches native PyTorch hooks at the TransformerLens hook names.
  The 70B loads in bfloat16 with **CPU offloading** (it does not fit in 80 GB on GPU alone).

---

## The three scripts

### 1. `dead_heads_distribution.py` — the main measurement
Computes the per-layer dead-head distribution for a model on a task suite and renders one bar
chart per task, plus a JSON with all raw numbers.

```bash
python3 dead_heads_distribution.py \
    --model meta-llama/Meta-Llama-3-70B \
    --device cuda --hf_token $HF_TOKEN \
    --suite serial_depth \
    --metric magnitude \
    --out graphs_out/serial_depth
```

Outputs, per task `i`: `<model>_task{i}_dead_heads.png` (bar chart) and one
`<model>_dead_heads.json` holding, for every prompt, the per-layer dead counts **and** the raw
per-head score matrix (so the data is never lost to a PNG).

### 2. `distribution_analysis.py` — head-set consistency study
Takes a `consistency`-suite JSON and asks *do same-category prompts activate the same heads?*
Emits activation-frequency maps, within/between-category overlap, a threshold-free consistency
measure, variance-explained, and core-head sets (see [distribution study](#consistency-study)).

```bash
python3 distribution_analysis.py \
    --json graphs_out/consistency/<model>_dead_heads.json \
    --tau 0.15 --core-theta 0.8 --out dist_out
```

### 3. `active_subgraph_dot.py` — the pipeline + graph renderer
Defines the model loaders, the metric functions, and all task suites, and can render the full
per-head computational graph (Mermaid / Graphviz DOT / PNG heatmap) of the active subgraph.
The other two scripts import their loaders and suites from here.

```bash
python3 active_subgraph_dot.py --model gpt2 --suite complexity_gradient --out graphs/gpt2
```

---

## How things are computed

### Dead vs. active heads

Each attention head `(ℓ, h)` gets a scalar **activity score** `s(ℓ,h)`. A head is **dead** for a
prompt when its score is below a fixed fraction of the strongest head *in its own layer*:

```
head (ℓ,h) is DEAD   ⇔   s(ℓ,h)  <  τ · max_h' s(ℓ,h')          with τ = 0.15 (default)
```

The threshold is **relative to the per-layer maximum**, so it adapts to each layer's scale.
(Note: with a relative threshold the top head in a layer is always active, so "all heads dead"
in a layer can only mean the scores are all zero — a useful failure signal.)

### The three activity metrics (`--metric`)

The per-head signal `s(ℓ,h)` can be computed three ways. All use `z`, the per-head attention
output captured at `blocks.{ℓ}.attn.hook_z` (shape `[batch, seq, n_heads, d_head]`).

| `--metric` | `s(ℓ,h)` | Needs backprop? | Runs on offloaded 70B? | Notes |
|---|---|---|---|---|
| **`attribution`** | `mean\|z · ∂logit/∂z\|` (gradient × activation) | **yes** | **no** | Identical to the original GPT-2/Pythia figures. Only works on models that fit fully on GPU. |
| **`contribution`** | `‖z_h · W_O_h‖` (residual-stream write norm) | no | **yes** | Attribution-like: weights the value vector by the output projection, discounting GQA-redundant heads. **Recommended for large models.** |
| **`magnitude`** | `‖z_h‖` (raw value norm) | no | yes | Crudest; ignores the output projection. |

**Why attribution can't run on the 70B:** it needs a backward pass, but `accelerate` offloads
each module's weights to CPU immediately after that module's forward — by backward time the
weights are gone, so no gradients flow and every score returns zero. The `contribution` metric
sidesteps this by computing `‖z_h · W_O_h‖` *inside* the module's forward (where `W_O` is
momentarily on-device). On the 70B, `contribution ≈ magnitude` (per-layer r ≈ 0.95–0.98), so
the choice between the two forward-only metrics doesn't change conclusions.

### Block-level magnitude ratios (in `active_subgraph_dot.py`)

For the coarse "is this whole block active" view, blocks are scored by
`r = ‖block_out‖ / ‖embedding_norm‖` evaluated at the final token — a block is skip-dominant
(inactive) when `r` is small.

### Derived geometry measures

From the per-layer dead array `d(ℓ)` (normalize depth to `u = ℓ/(L−1) ∈ [0,1]`):

- **Recruited width** `Ŵ` = mean **active** heads per layer (`N_heads − d(ℓ)`, averaged).
- **Recruited depth / active frontier** `D̂` = a front-edge quantile of the dead-mass
  distribution: `q₂₅` = the depth by which the first quarter of dead mass has accumulated.
  (The centroid `q₅₀` is a poor frontier measure because it is anchored by the always-dead
  readout region at the back; `q₂₅` tracks the *front edge* where recruitment happens.)

---

## Task suites (with exact prompts)

Select with `--suite <name>`. Every suite is a list of `(prompt, label)` pairs. The factorial
and consistency suites encode their coordinates in the label (`D{d}W{w}`, `{category}#{i}`) so
downstream analysis can regress on them.

### Descriptive suites

**`complexity_gradient`** — 8 heterogeneous tasks spanning a rough difficulty range:

| Label | Prompt |
|---|---|
| Lexical | `The dog barked loudly.` |
| Subject-verb agreement | `The keys to the cabinet are on the table. The keys` |
| 1-hop factual | `The capital of France is` |
| Arithmetic | `17 plus 28 equals` |
| 2-hop reasoning | `Alice is the mother of Bob. Bob is the mother of Carol. Alice's grandchild is` |
| 3-hop reasoning | `Alice is the parent of Bob. Bob is the parent of Carol. Carol is the parent of Dana. Alice's great-grandchild is` |
| Logical syllogism | `All mammals breathe air. Dolphins are mammals. Therefore, dolphins` |
| Analogy | `Paris is to France as Berlin is to` |

**`deep_chains`** — a clean serial-reasoning ladder, 1-hop … 6-hop, same family-tree template:
```
1-hop: "Alice is the mother of Bob. Alice's child is"
2-hop: "Alice is the mother of Bob. Bob is the mother of Carol. Alice's grandchild is"
   …
6-hop: "A is the parent of B. … F is the parent of G. A's descendant six generations down is"
```
Adds one composition step per rung, but the **prompt gets longer** with depth (a confound the
factorial suite fixes).

Also available: `quick`, `syntax`, `arithmetic`, `reasoning`, `world_knowledge`, `surface`,
`mixed` (see `TASK_SUITES` in `active_subgraph_dot.py`).

### Factorial suites — the (D, W) dissociation

These vary **serial depth `D`** and **parallel width `W`** independently, to test the headline
hypothesis. Labels are `D{d}W{w}`.

**`serial_depth`** (pure serial: `W=1`, `D=1…6`). A fixed 5-cycle pointer-chase; only the step
count changes, so **prompt length is constant across `D`** — recruitment cannot be tracking token
count. This is the cleanest depth manipulation.
```
"Rules: 0 goes to 2. 1 goes to 4. 2 goes to 1. 3 goes to 0. 4 goes to 3. Start at 0. Take 3 steps. You end at"   → 4
```

**`parallel_width`** (pure parallel: `D=1`, `W=1…6`). Find the unique binding among `W` items.
```
"Berlin is in Germany. Cairo is in Egypt. Lima is in Peru. The city in Egypt is"   → Cairo
```
Prompt length grows with `W`.

**`parallel_width_lm`** — same, but padded with neutral filler to **constant length** (controls
for the length growth of the width axis). *This control is necessary* — see Results.

**`dw_grid`** — crossed 3×3 (`D,W ∈ {1,3,5}`): `W` parallel depth-`D` pointer chains, answer =
largest endpoint. Tests additivity/interaction.
```
"Rules: … Take 3 steps from each of these starts: 0, 1, 2. The largest value you reach is"
```

**`dw_factorial`** — the two pure axes concatenated, for a one-shot run.

### <a name="consistency-study"></a>Consistency suite — the distribution study

**`consistency`** — 5 categories × 30 **paraphrased** instances (150 prompts). Each category
holds its computation fixed but varies entities, wording, and order, so any shared active-head
core reflects the *computation*, not the template. Labels are `{category}#{i}`.

| Category | Computation | Example instance |
|---|---|---|
| `fact1hop` | single-hop factual recall | `Portugal's capital is` |
| `reason2hop` | 2-hop grandparent | `Kate is Henry's mother. Henry is Iris's mother. The grandchild of Kate is` |
| `arith` | two-digit addition | `The sum of 37 and 30 is` |
| `serial3` | depth-3 pointer chase | `Rules: 0 goes to 2. 2 goes to 4. … Start at 1. Take 3 steps. You end at` |
| `parallel4` | width-4 find-match | `Cairo is in Egypt. Madrid is in Spain. Lima is in Peru. Tokyo is in Japan. The city in Egypt is` |

`distribution_analysis.py` then computes, per category:
- **`f_c(ℓ,h)`** = fraction of category prompts that activate head `(ℓ,h)` — the
  **activation-frequency map**; its histogram is expected to be **trimodal** (never-active /
  variable periphery / a core spike near 1).
- **within- vs between-category Jaccard** of active-head sets (predict within ≫ between).
- **rank consistency** = per-layer Spearman of head scores across prompts — a *threshold-free*
  consistency measure.
- **η²** = fraction of per-head activity variance explained by category.
- **permutation-null z-score** for the within-Jaccard.
- **core set + geometry** = heads with `f_c ≥ θ` (default 0.8), plus its depth-span and
  heads-per-layer (to line up against the category's `D`/`W`).

---

## Current results

All numbers below are from runs archived under `johnny_results/`, `abdulla_results/`, and
`graphs_8b_0512/`.

### 1. Recruitment (vertical axis) — confirmed
As a task gets more demanding, the model activates more heads (dead-head count falls). Cleanest
on the reasoning ladder: on GPT-2-xl the `deep_chains` dead count falls **228 → 97** from 1-hop
to 6-hop; the first-hop step (1→2) recruits 20–55 % of all idle heads in *every* model.

### 2. Scale dilation — confirmed
The recruitable reserve grows with model size. GPT-2 idles **8 % → 10 % → 19 %** of heads
(medium → large → xl) on the easiest task; Llama-3-70B idles **~50 %** even on hard tasks — so
its dynamic range is compressed (a huge reserve). This makes the 70B a poor instrument for
subtle effects.

### 3. Frontier (horizontal axis) — confirmed on smaller models, null on 70B
On the exact **Llama-3-8B** reasoning subset, the active frontier extends deeper with reasoning
depth: `r(hops, q₂₅) = +0.96`, `r(hops, centroid) = +1.00`, early-layer dead falls 92 → 42
across 1/2/3-hop. Pooled over the three largest GPT-2/Pythia models, `r(hops, q₂₅) = +0.70`.

### 4. (D, W) factorial on Llama-3-70B — width half confirmed, depth half null
Controlled run varying `D` and `W` independently (forward-only magnitude metric):
- **Width recruits heads:** grid fit `active-heads/layer ≈ 32.3 − 0.14·D + 0.25·W` — a clean
  positive width coefficient, ~zero depth coefficient; length-matched parallel ladder
  `r(W, active/layer) = +0.81`.
- **Depth does *not* recruit heads**, and shows **no frontier movement** on the 70B (front/
  centroid/back quantiles shift `<1` layer over `D=1…6` — noise). Read as an instrument problem
  (the 70B's enormous reserve), not a refutation; the depth axis needs the low-reserve small
  models plus accuracy measurements.
- **The length control is necessary:** the *raw* parallel ladder trends the *wrong* way (dead
  rises with `W`); holding prompt length constant flips it to the predicted direction.

### 5. Metric & reproducibility checks
- **Metric-robust:** on the 70B, `contribution` (`‖z·W_O‖`) and `magnitude` (`‖z‖`) agree to
  per-layer r ≈ 0.95–0.98 — the ~55 % dead fraction is real, not a metric artifact.
- **Deterministic:** re-running the same suite/metric reproduces every number exactly (the
  forward metric has zero run-to-run variance).

### 6. Distribution study — tooling built & validated, real run pending
`distribution_analysis.py` is validated on synthetic planted-core data (recovers the planted
core exactly; within-Jaccard 0.70 vs between 0.20; η² 0.71). Awaiting a real `consistency` run.

---

## Outputs & data folders

| Folder | Contents |
|---|---|
| `graphs_8b_0512/` | Llama-3-8B, full per-head **attribution** graphs (8 tasks) — the exact reference figures. |
| `abdulla_results/` | GPT-2 (medium/large/xl) + Pythia (70M/160M/410M) sweep across `complexity_gradient` and `deep_chains` (dead-head PNGs). |
| `johnny_results/` | Llama-3-70B runs: `complexity_gradient`, the `(D,W)` factorial (`graphs_70b_factorial/`), and dated snapshots. |
| `paper_manuscript/` | The write-up (`dead_head_recruitment.tex` / `.pdf`) and its figures. |

Each `dead_heads_distribution.py` run also writes a `*_dead_heads.json` next to the PNGs — the
authoritative numeric record (per-layer dead counts + per-head scores per prompt).

---

## The paper

`paper_manuscript/dead_head_recruitment.pdf` — *"Complexity-Graded Head Recruitment: Dead
Attention Heads as a Behavioural Probe of Task Demand."* Documents the recruitment and frontier
axes, the scale-dilation effect, the `(D,W)` factorial dissociation, the complete dataset, and
the experimental roadmap. Build with `pdflatex dead_head_recruitment.tex` (×3, for refs +
longtables).

---

## Caveats

- **Absolute dead fractions are not comparable across metrics.** The GPT-2/Pythia figures use
  `attribution` (~8–19 % dead); the 70B uses forward-only `magnitude`/`contribution` (~50–60 %).
  These answer different questions ("relevant to this logit?" vs. "doing anything?"). For a
  cross-scale comparison, use **one** metric everywhere — `magnitude` is the only one that runs
  on both small (TransformerLens) and offloaded-70B models.
- **Scale dilation** makes the 70B the *worst* instrument for the subtle depth/frontier effects;
  run the factorial + consistency suites on GPT-2/Pythia for a clearer signal.
- **No accuracy coupling yet.** Gold answers are stashed in the factorial/consistency suites
  (`_answers`) but not yet scored; without them we cannot confirm the model actually performs a
  task serially. This is the top item on the roadmap.
