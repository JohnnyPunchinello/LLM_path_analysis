#!/usr/bin/env python3
"""
dead_heads_distribution.py
==========================

Compute and plot the *distribution of dead attention heads across layers* for a
model + task suite, producing one bar chart per task in the same style as the
`*_dead_heads.png` figures analysed previously.

WHY A SEPARATE SCRIPT
---------------------
`active_subgraph_dot.py` measures head activity via gradient x activation
attribution.  On a 70B model loaded in bfloat16 with CPU offloading, a full
backward pass through 80 offloaded layers is impractical (and, before the
recent dtype fix, produced all-zero scores).  This script instead uses a
*gradient-free* head-activity measure that needs only a forward pass:

    activity(l, h) = || z_{l,h} ||_2        (per-head attention output norm)
    head (l,h) is DEAD  iff  activity(l,h) < tau * max_h' activity(l,h')

where z is the per-head value-weighted output captured at `blocks.l.attn.hook_z`
(shape [batch, seq, n_heads, d_head]).  This is robust, fast, and works
identically on TransformerLens models (small) and the native-hook `_HFHookedModel`
wrapper (large / 70B).

It also DUMPS THE RAW NUMBERS (per-layer dead counts and per-head norms) to JSON
so the underlying data is preserved and never has to be recovered from a PNG.

USAGE
-----
    python3 dead_heads_distribution.py \
        --model meta-llama/Meta-Llama-3-70B \
        --device cuda \
        --hf_token hf_xxx \
        --suite complexity_gradient \
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

# Reuse the (already-working) loader and task suites from the main pipeline.
from active_subgraph_dot import load_model, TASK_SUITES


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement
# ─────────────────────────────────────────────────────────────────────────────
def dead_heads_per_layer(
    model,
    tokens: torch.Tensor,
    rel_threshold: float = 0.15,
    position: str = "last",
):
    """
    Return (dead_counts, per_head_norm):
      dead_counts    : int ndarray [n_layers]  — # heads below threshold per layer
      per_head_norm  : float ndarray [n_layers, n_heads] — the raw activity signal

    A head is dead when its attention-output L2 norm is below `rel_threshold`
    times the strongest head in the same layer.

    position : "last"  → evaluate at the final (prediction) token
               "mean"  → average the per-head norm over all sequence positions
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
                    help="Head dead if norm < threshold x layer-max (default 0.15).")
    ap.add_argument("--position", default="last", choices=["last", "mean"],
                    help="Evaluate at final token (default) or mean over sequence.")
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
    print(f"  n_layers={n_layers}  n_heads={n_heads}  "
          f"metric=||z_head|| ({args.position})  threshold={args.threshold}\n")

    dump = {"model": args.model, "suite": args.suite,
            "n_layers": n_layers, "n_heads": n_heads,
            "metric": f"per-head z-norm ({args.position})",
            "threshold": args.threshold, "tasks": []}

    for i, (text, label) in enumerate(zip(tasks, labels)):
        print(f"  [{i+1}/{len(tasks)}] {label!r}: {textwrap.shorten(text, 50)}")
        tokens = model.to_tokens(text)
        if tokens.shape[-1] > 512:
            tokens = tokens[:, -512:]
        dead, norms = dead_heads_per_layer(
            model, tokens, rel_threshold=args.threshold, position=args.position)
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
            "per_head_norm": norms.tolist(),   # raw signal, preserved
        })

    json_path = out_dir / f"{safe_name}_dead_heads.json"
    json_path.write_text(json.dumps(dump, indent=1))
    print(f"\nRaw data → {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
