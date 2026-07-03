#!/usr/bin/env python3
"""
dead_heads_distribution.py
==========================

Compute and plot the *distribution of dead attention heads across layers* for a
model + task suite, producing one bar chart per task in the same style as the
`*_dead_heads.png` figures analysed previously.

TWO METRICS
-----------
A head (l,h) is DEAD iff  activity(l,h) < tau * max_h' activity(l,h').  The
activity signal is selectable:

  --metric attribution   gradient x activation attribution of the per-head output
      toward the target logit.  *Identical* to the criterion behind the original
      GPT-2 / Pythia *_dead_heads figures (delegates to `compute_per_head_scores`
      + `_active_heads`).  Needs a BACKWARD pass, so it works only on models that
      fit fully on GPU.  It CANNOT run on a CPU-offloaded 70B: accelerate offloads
      each module's weights immediately after its forward, so by backward time
      the weights are gone and no activation gradients flow — every score comes
      back zero and every head reads as "dead".  The script aborts in that case.

  --metric contribution  (DEFAULT)  forward-only per-head write norm
      ||z_{l,h} @ W_O_h|| — the size of the head's actual contribution to the
      residual stream.  Attribution-like (it weights the value vector by the
      output projection, discounting GQA-redundant heads), but needs only a
      forward pass, so it works on the offloaded 70B.  Recommended for large
      models.

  --metric magnitude     crudest forward-only measure: raw per-head value norm
      ||z_{l,h}||.  Ignores the output projection, so it over-counts dead heads
      under grouped-query attention.

z is captured at `blocks.l.attn.hook_z` (shape [batch, seq, n_heads, d_head]);
both metrics work on TransformerLens models (small) and the native-hook
`_HFHookedModel` wrapper (large / 70B).

It DUMPS THE RAW NUMBERS (per-layer dead counts and per-head scores) to JSON so
the underlying data is preserved and never has to be recovered from a PNG.

USAGE
-----
    python3 dead_heads_distribution.py \
        --model meta-llama/Meta-Llama-3-70B \
        --device cuda \
        --hf_token hf_xxx \
        --suite complexity_gradient \
        --metric attribution \
        --out graphs_70b_deadheads

Outputs, per task i:
    <out>/<model>_task{i}_dead_heads.png    bar chart (dead heads per layer)
    <out>/<model>_dead_heads.json           all numeric data for every task
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import torch

# Reuse the (already-working) loader, task suites, and — for the attribution
# metric — the exact per-head scoring used to produce the original figures.
from active_subgraph_dot import (
    load_model, TASK_SUITES, compute_per_head_scores, _active_heads,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — ATTRIBUTION (identical to the original *_dead_heads figures)
# ─────────────────────────────────────────────────────────────────────────────
def dead_heads_attribution(
    model,
    tokens: torch.Tensor,
    rel_threshold: float = 0.15,
    target_pos: int = -1,
):
    """
    Old attribution metric.  A head's activity is the gradient x activation
    attribution of its per-head output z toward the target logit:

        score(l,h) = mean_{seq,d_head} | z_{l,h} * d(logit)/d z_{l,h} |

    and it is DEAD when score(l,h) < rel_threshold x max_h' score(l,h').

    This delegates to `compute_per_head_scores` and `_active_heads` from the
    main pipeline, so the criterion is byte-for-byte the same as the GPT-2 /
    Pythia figures.  Needs a backward pass; on a CPU-offloaded 70B this is
    slow (minutes per task) but works now that the anchor-dtype bug is fixed.

    Returns (dead_counts [n_layers], head_scores [n_layers, n_heads]).
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    head_scores, _ = compute_per_head_scores(model, tokens, target_pos=target_pos)

    dead = np.zeros(n_layers, dtype=int)
    for l in range(n_layers):
        active = _active_heads(head_scores, l, rel_threshold)  # list[bool]
        dead[l] = n_heads - int(sum(active))
    return dead, head_scores


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — MAGNITUDE (gradient-free fallback; forward pass only)
# ─────────────────────────────────────────────────────────────────────────────
def dead_heads_magnitude(
    model,
    tokens: torch.Tensor,
    rel_threshold: float = 0.15,
    position: str = "last",
):
    """
    Gradient-free alternative.  Head activity is the per-head attention-output
    L2 norm ||z_{l,h}||; dead when below rel_threshold x layer-max.  Only needs
    a forward pass, so it is robust on very large / offloaded models where a
    backward pass is impractical.

    position : "last"  → evaluate at the final (prediction) token
               "mean"  → average the per-head norm over all sequence positions

    Returns (dead_counts [n_layers], per_head_norm [n_layers, n_heads]).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    store: dict = {}
    fwd_hooks = []
    for l in range(n_layers):
        def _cap(act, hook, ll=l):
            # act: [batch, seq, n_heads, d_head]; capture only (return None)
            store[ll] = act.detach()
            return None
        fwd_hooks.append((f"blocks.{l}.attn.hook_z", _cap))

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type="logits")

    dead = np.zeros(n_layers, dtype=int)
    per_head_norm = np.zeros((n_layers, n_heads), dtype=np.float32)

    for l in range(n_layers):
        z = store.get(l)
        if z is None:
            # Layer's hook never fired (unrecognised module); mark all dead.
            dead[l] = n_heads
            continue
        z = z.float()                       # [1, seq, n_heads, d_head]
        norms = z[0].norm(dim=-1)           # [seq, n_heads]
        if position == "mean":
            v = norms.mean(dim=0)           # [n_heads]
        else:                               # "last"
            v = norms[-1]                   # [n_heads]
        v = v.cpu().numpy()
        # Guard against a captured head count that differs from cfg.n_heads
        m = min(len(v), n_heads)
        per_head_norm[l, :m] = v[:m]
        mx = float(v.max()) if v.size else 0.0
        dead[l] = int((v < rel_threshold * mx).sum()) if mx > 0 else n_heads

    return dead, per_head_norm


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — CONTRIBUTION (forward-only, attribution-like; recommended for 70B)
# ─────────────────────────────────────────────────────────────────────────────
def dead_heads_contribution(
    model,
    tokens: torch.Tensor,
    rel_threshold: float = 0.15,
    position: str = "last",
):
    """
    Forward-only head activity = the L2 norm of the head's actual write into the
    residual stream, ||z_{l,h} @ W_O_h||, where W_O_h is head h's slice of the
    attention output projection.  Dead when below rel_threshold x layer-max.

    Why this and not raw ||z||: a head can have a large value vector z but a small
    output-projection footprint (common under grouped-query attention, where many
    query heads in a group are near-redundant).  Weighting by W_O measures the
    head's true contribution to the stream, so it tracks gradient x activation
    attribution far more closely than ||z|| while needing only a forward pass —
    which is essential on a CPU-offloaded 70B where a backward pass cannot run.

    Implemented by wrapping each o_proj's real forward (`_old_forward` under
    accelerate, else `forward`), which is the only point where W_O is materialised
    on-device for a CPU-offloaded layer — a standard forward_pre_hook fires too
    early (before accelerate loads the weight) and sees a meta tensor.  Requires
    the `_HFHookedModel` wrapper (large HF models).

    Returns (dead_counts [n_layers], per_head_contrib [n_layers, n_heads]).
    """
    hf = getattr(model, "_model", None)
    if hf is None or not hasattr(model, "_layers"):
        raise RuntimeError(
            "The 'contribution' metric needs the _HFHookedModel wrapper "
            "(large HF models). For small TransformerLens models use "
            "--metric attribution.")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    layers = model._layers()

    store: dict = {}
    patched: list = []   # (module, attr_name, original_fn)

    def _make_wrapper(orig_fn, module, ll):
        def _wrapper(*a, **kw):
            # Runs inside accelerate's execution window → module.weight is live.
            try:
                x = a[0] if a else next(iter(kw.values()))
                W = module.weight
                if x.dim() == 3 and W.device.type != "meta":
                    b, s, _ = x.shape
                    xh = x.reshape(b, s, n_heads, d_head)[0].float()     # [s, nh, dh]
                    Wr = W.reshape(W.shape[0], n_heads, d_head).float()  # [hidden, nh, dh]
                    contrib = torch.einsum("shd,ohd->sho", xh, Wr)       # [s, nh, hidden]
                    store[ll] = contrib.norm(dim=-1).detach().cpu()      # [s, nh]
            except Exception:
                pass   # never let measurement break the forward
            return orig_fn(*a, **kw)
        return _wrapper

    for l in range(n_layers):
        attn = model._get_attn(layers[l])
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None:
            continue
        # Wrap the point where the (possibly offloaded) weight is live.
        attr = "_old_forward" if hasattr(o_proj, "_old_forward") else "forward"
        orig = getattr(o_proj, attr)
        setattr(o_proj, attr, _make_wrapper(orig, o_proj, l))
        patched.append((o_proj, attr, orig))

    try:
        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=None, return_type="logits")
    finally:
        for o_proj, attr, orig in patched:
            setattr(o_proj, attr, orig)

    dead = np.zeros(n_layers, dtype=int)
    per_head = np.zeros((n_layers, n_heads), dtype=np.float32)
    for l in range(n_layers):
        norms = store.get(l)
        if norms is None:
            dead[l] = n_heads
            continue
        v = (norms.mean(0) if position == "mean" else norms[-1]).numpy()
        m = min(len(v), n_heads)
        per_head[l, :m] = v[:m]
        mx = float(v.max()) if v.size else 0.0
        dead[l] = int((v < rel_threshold * mx).sum()) if mx > 0 else n_heads

    return dead, per_head


# ─────────────────────────────────────────────────────────────────────────────
# Plot (matches the *_dead_heads.png style)
# ─────────────────────────────────────────────────────────────────────────────
def plot_dead_heads(dead, n_heads, model_name, label, text, out_png, rel_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = len(dead)
    fig, ax = plt.subplots(figsize=(max(14, n_layers * 0.30), 5))
    xs = list(range(n_layers))

    ax.bar(xs, dead, color="#808080", width=0.8,
           label=f"Dead heads (< {rel_threshold:g} × max)")
    ax.axhline(n_heads, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"Total heads per layer ({n_heads})")
    for x, v in zip(xs, dead):
        if v > 0:
            ax.text(x, v + max(0.2, n_heads * 0.01), str(int(v)),
                    ha="center", va="bottom", fontsize=7)

    short = textwrap.shorten(text, width=60, placeholder=" ...")
    ax.set_title(f"Dead heads per layer — {model_name}\n"
                 f'Task: {label} — "{short}"', fontsize=11)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Number of dead attention heads")
    step = max(1, n_layers // 24)
    ax.set_xticks(range(0, n_layers, step))
    ax.set_xticklabels(range(0, n_layers, step), fontsize=7)
    ax.set_ylim(0, n_heads * 1.08)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Distribution of dead attention heads across layers.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--suite", default="complexity_gradient",
                    choices=list(TASK_SUITES.keys()))
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Custom prompts (overrides --suite).")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default="graphs_deadheads",
                    help="Output directory.")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="Head dead if score < threshold x layer-max (default 0.15).")
    ap.add_argument("--metric", default="contribution",
                    choices=["attribution", "contribution", "magnitude"],
                    help="attribution: gradient x activation, identical to the "
                         "original GPT-2/Pythia figures — needs a backward pass, "
                         "so it CANNOT run on a CPU-offloaded 70B (accelerate "
                         "offloads weights after each forward). Use on models "
                         "that fit fully on GPU.  contribution (DEFAULT): "
                         "forward-only ||z_h.W_O_h||, attribution-like, works on "
                         "the offloaded 70B, discounts GQA-redundant heads.  "
                         "magnitude: crudest, forward-only ||z_h||.")
    ap.add_argument("--position", default="last", choices=["last", "mean"],
                    help="contribution/magnitude metrics: final token (default) "
                         "or mean over sequence.")
    args = ap.parse_args()

    if args.tasks:
        tasks = args.tasks
        labels = args.labels or [f"task{i}" for i in range(len(tasks))]
    else:
        suite = TASK_SUITES[args.suite]
        tasks, labels = suite["tasks"], suite["labels"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_")

    print("=" * 60)
    print(f"Dead-head distribution: {args.model}")
    print("=" * 60)
    model = load_model(args.model, device=args.device, hf_token=args.hf_token)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    metric_str = {
        "attribution":  "gradient x activation attribution",
        "contribution": f"per-head ||z.W_O|| contribution ({args.position})",
        "magnitude":    f"per-head z-norm ({args.position})",
    }[args.metric]
    print(f"  n_layers={n_layers}  n_heads={n_heads}  "
          f"metric={metric_str}  threshold={args.threshold}\n")
    if args.metric == "attribution":
        print("  NOTE: attribution needs a backward pass. It works on models that\n"
              "  fit fully on GPU, but CANNOT run on a CPU-offloaded 70B — the run\n"
              "  will abort if scores come back all-zero. Use --metric contribution\n"
              "  for the 70B.\n")

    dump = {"model": args.model, "suite": args.suite,
            "n_layers": n_layers, "n_heads": n_heads,
            "metric": metric_str, "threshold": args.threshold, "tasks": []}

    for i, (text, label) in enumerate(zip(tasks, labels)):
        print(f"  [{i+1}/{len(tasks)}] {label!r}: {textwrap.shorten(text, 50)}")
        tokens = model.to_tokens(text)
        if tokens.shape[-1] > 512:
            tokens = tokens[:, -512:]

        if args.metric == "attribution":
            dead, raw = dead_heads_attribution(
                model, tokens, rel_threshold=args.threshold)
            # Hard-stop instead of emitting all-dead garbage: all scores zero
            # means the backward pass produced no gradients (e.g. CPU-offloaded
            # model).  Abort with an actionable message.
            if float(np.max(raw)) <= 0.0:
                raise SystemExit(
                    "\nERROR: attribution scores are all zero — the backward pass "
                    "produced no gradients.\nThis happens on CPU-offloaded models "
                    "(the 70B): accelerate offloads each module's weights right "
                    "after its forward,\nso there are no weights left for backward. "
                    "Re-run with:  --metric contribution\n")
        elif args.metric == "contribution":
            dead, raw = dead_heads_contribution(
                model, tokens, rel_threshold=args.threshold,
                position=args.position)
        else:
            dead, raw = dead_heads_magnitude(
                model, tokens, rel_threshold=args.threshold,
                position=args.position)

        total = int(dead.sum())
        print(f"       total dead = {total}/{n_layers * n_heads}  "
              f"({100 * total / (n_layers * n_heads):.1f}%)")

        png = out_dir / f"{safe_name}_task{i}_dead_heads.png"
        plot_dead_heads(dead, n_heads, args.model, label, text, str(png),
                        args.threshold)
        print(f"       → {png}")

        dump["tasks"].append({
            "index": i, "label": label, "text": text,
            "dead_per_layer": dead.tolist(),
            "total_dead": total,
            "per_head_score": raw.tolist(),   # raw signal, preserved
        })

    json_path = out_dir / f"{safe_name}_dead_heads.json"
    json_path.write_text(json.dumps(dump, indent=1))
    print(f"\nRaw data → {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
