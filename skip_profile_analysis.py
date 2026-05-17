#!/usr/bin/env python3
"""
skip_profile_analysis.py
========================
Cross-model, cross-task active-subgraph skip profiler.

Runs every combination of (model × suite × task), extracts quantitative
metrics from the residual/skip-connection profile, and writes:

  <out>.csv              — one row per (model, suite, task)
  <out>_heatmap.png      — FFN-active heatmap: rows=tasks, cols=normalised depth
  <out>_horizon.png      — compute-horizon % by task, grouped bars per model
  <out>_scatter.png      — fragmentation vs late-attn scatter (colour = task type)
  <out>_lines.png        — compute horizon vs hop-count lines (deep_chains suite)

Metrics extracted per (model, task)
------------------------------------
  k                   active edges at 90% nucleus coverage
  compute_horizon_pct depth of last active FFN, normalised [0, 1]
  skip_com_pct        centre-of-mass of FFN-skip layers, normalised [0, 1]
  fragmentation       # contiguous FFN-skip runs  (1 = clean tail, 3 = multi-stage)
  tail_skip_pct       where the terminal skip zone begins, normalised [0, 1]
  late_attn_frac      mean head-fraction active in last 3 layers (0=chain, 1=logic)
  attn_cutoff_pct     normalised depth where attention permanently falls silent

Usage
-----
  # Quick cross-model comparison on reasoning + deep_chains
  python skip_profile_analysis.py \\
      --models gpt2 gpt2-medium EleutherAI/pythia-160m \\
      --suites reasoning deep_chains \\
      --device cuda --out results/skip

  # Broad survey: many models, all suites
  python skip_profile_analysis.py \\
      --models gpt2 gpt2-medium gpt2-large \\
              EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m \\
      --suites mixed surface deep_chains reasoning \\
      --device cuda --out results/skip_broad

Available suites: quick, complexity_gradient, syntax, arithmetic, reasoning,
                  world_knowledge, deep_chains, surface, mixed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from active_subgraph_dot import (
    load_model, compute_per_head_scores,
    TASK_SUITES, _active_heads,
)
from path_analyzer import select_active_edges_by_mass_coverage

MASS_COVERAGE  = 0.90
HEAD_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(
    active_attn   : List[bool],
    active_mlp    : List[bool],
    head_scores   : np.ndarray,   # [n_layers, n_heads]
    n_heads       : int,
    k             : int,
    head_threshold: float = HEAD_THRESHOLD,
) -> Dict[str, Any]:
    """Compute all skip-profile metrics for one (model, task) pair."""
    n   = len(active_mlp)
    ffn = [int(b) for b in active_mlp]

    # Per-layer active head counts
    hc = []
    for l in range(n):
        if active_attn[l]:
            hc.append(sum(_active_heads(head_scores, l, head_threshold)))
        else:
            hc.append(0)

    # ── Compute horizon: last layer with FFN ✓ ───────────────────────────
    horizon = -1
    for l in range(n - 1, -1, -1):
        if ffn[l]:
            horizon = l
            break

    # ── Skip layers and their centre-of-mass ────────────────────────────
    skips   = [l for l in range(n) if not ffn[l]]
    com     = float(np.mean(skips)) if skips else float(n - 1)
    com_pct = com / max(n - 1, 1)

    # ── Fragmentation: # contiguous FFN-skip runs ────────────────────────
    frag, in_run = 0, False
    for b in ffn:
        if not b:
            if not in_run:
                frag += 1
                in_run = True
        else:
            in_run = False

    # ── Terminal skip zone start ─────────────────────────────────────────
    tail = n
    for l in range(n - 1, -1, -1):
        if ffn[l]:
            break
        tail = l

    # ── Late-layer attention (last 3 layers) ────────────────────────────
    w          = min(3, n)
    late_tot   = sum(hc[n - w:])
    late_frac  = late_tot / (n_heads * w) if n_heads > 0 else 0.0

    # ── Attention cutoff: first layer where attn stays silent ────────────
    cutoff = n
    for l in range(n):
        tail_max = max((hc[ll] / n_heads for ll in range(l, n)), default=0)
        if tail_max < 0.20:
            cutoff = l
            break

    return dict(
        k                   = k,
        n_ffn_active        = sum(ffn),
        n_attn_layers       = sum(1 for c in hc if c > 0),
        compute_horizon     = horizon,
        compute_horizon_pct = round((horizon + 1) / n if horizon >= 0 else 0.0, 4),
        skip_com_pct        = round(com_pct, 4),
        fragmentation       = frag,
        tail_skip_pct       = round(tail / n, 4),
        late_attn_frac      = round(float(late_frac), 4),
        attn_cutoff_pct     = round(cutoff / n, 4),
        ffn_profile         = json.dumps(ffn),
        attn_profile        = json.dumps([round(c / n_heads, 3) for c in hc]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def analyse_model(
    model_name   : str,
    suites       : List[str],
    device       : str,
    hf_token     : Optional[str],
    mass_coverage: float = MASS_COVERAGE,
) -> List[Dict]:
    """Load a model once and process all (suite × task) pairs."""
    print(f"\n{'='*62}\nModel: {model_name}\n{'='*62}")
    try:
        model = load_model(model_name, device=device, hf_token=hf_token)
        model.eval()
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return []

    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    dev      = next(model.parameters()).device
    mname    = getattr(model.cfg, "model_name", model_name)
    print(f"  {n_layers}L × {n_heads}H  device={dev}")

    rows: List[Dict] = []
    for suite_name in suites:
        if suite_name not in TASK_SUITES:
            print(f"  Unknown suite '{suite_name}' — skipping")
            continue
        suite  = TASK_SUITES[suite_name]
        tasks  = suite["tasks"]
        labels = suite["labels"]

        for i, (text, label) in enumerate(zip(tasks, labels)):
            print(f"  [{suite_name}  {i+1}/{len(tasks)}]  {label!r}")
            try:
                tokens = model.to_tokens(text)
                if tokens.shape[-1] > 512:
                    tokens = tokens[:, -512:]
                tokens = tokens.to(dev)

                hs, ms = compute_per_head_scores(model, tokens)
                attn_layer_scores = hs.max(axis=1)
                act_a, act_m, eps, k = select_active_edges_by_mass_coverage(
                    attn_layer_scores, ms, mass_fraction=mass_coverage,
                )
                metrics = extract_metrics(act_a, act_m, hs, n_heads, k)

                rows.append({
                    "model"      : mname,
                    "suite"      : suite_name,
                    "task_label" : label,
                    "task_text"  : textwrap.shorten(text, 90),
                    "n_layers"   : n_layers,
                    "n_heads"    : n_heads,
                    **{fk: fv for fk, fv in metrics.items()
                       if fk not in ("ffn_profile", "attn_profile")},
                    "ffn_profile" : metrics["ffn_profile"],
                    "attn_profile": metrics["attn_profile"],
                })
                print(f"    k={k:3d}  horizon={metrics['compute_horizon_pct']:.2f}"
                      f"  frag={metrics['fragmentation']}"
                      f"  late_attn={metrics['late_attn_frac']:.2f}")
            except Exception as exc:
                import traceback
                print(f"    ERROR: {exc}")
                traceback.print_exc()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "model", "suite", "task_label", "task_text", "n_layers", "n_heads",
    "k", "n_ffn_active", "n_attn_layers",
    "compute_horizon", "compute_horizon_pct",
    "skip_com_pct", "fragmentation", "tail_skip_pct",
    "late_attn_frac", "attn_cutoff_pct",
    "ffn_profile", "attn_profile",
]

def save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV → {path}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Task-type colour map
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_COLOURS = {
    "surface"     : "#aec7e8",
    "trivial"     : "#aec7e8",
    "number seq"  : "#aec7e8",
    "factual"     : "#ffbb78",
    "capital"     : "#ffbb78",
    "author"      : "#ffbb78",
    "chemical"    : "#ffbb78",
    "historical"  : "#ffbb78",
    "astronomy"   : "#ffbb78",
    "arithmetic"  : "#ffd700",
    "addition"    : "#ffd700",
    "multiplication": "#ffd700",
    "division"    : "#ffd700",
    "percentage"  : "#ffd700",
    "word problem": "#ffd700",
    "trivial arith": "#ffd700",
    "1-hop"       : "#d62728",
    "2-hop"       : "#ff7f0e",
    "3-hop"       : "#e377c2",
    "4-hop"       : "#9467bd",
    "5-hop"       : "#8c564b",
    "6-hop"       : "#7f7f7f",
    "syllogism"   : "#17becf",
    "negation"    : "#1f77b4",
    "counterfact" : "#2ca02c",
    "analogy"     : "#98df8a",
    "syntax"      : "#bcbd22",
    "subject-verb": "#bcbd22",
    "agreement"   : "#bcbd22",
    "relative"    : "#bcbd22",
    "embedded"    : "#bcbd22",
    "either-or"   : "#bcbd22",
}

def _task_colour(label: str) -> str:
    lo = label.lower()
    for key, c in _TYPE_COLOURS.items():
        if key in lo:
            return c
    return "#cccccc"


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: FFN-active heatmap (one panel per model)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(rows: List[Dict], out_path: Path) -> None:
    """
    Heatmap: rows = tasks, cols = normalised layer depth (20 bins).
    Blue = FFN active, white = FFN skip.
    One subplot per model.
    """
    models   = list(dict.fromkeys(r["model"] for r in rows))
    all_tasks = list(dict.fromkeys(r["task_label"] for r in rows))
    N_BINS   = 20
    n_m      = len(models)

    fig_h = max(5, len(all_tasks) * 0.45 + 2)
    fig, axes = plt.subplots(
        1, n_m, figsize=(5.5 * n_m, fig_h), squeeze=False
    )
    fig.suptitle(
        "FFN-active profile per layer depth\n"
        "(blue = FFN active, white = residual skip)",
        fontsize=12, y=1.01,
    )
    cmap = LinearSegmentedColormap.from_list("ffn", ["#f7fbff", "#08519c"])

    for col, mname in enumerate(models):
        ax    = axes[0][col]
        mrows = [r for r in rows if r["model"] == mname]
        if not mrows:
            ax.set_visible(False)
            continue

        # Build per-task ordering
        task_order = list(dict.fromkeys(r["task_label"] for r in mrows))
        n_t        = len(task_order)
        grid       = np.zeros((n_t, N_BINS))

        for r in mrows:
            ti  = task_order.index(r["task_label"])
            ffn = json.loads(r["ffn_profile"])
            n   = len(ffn)
            for l, v in enumerate(ffn):
                b = min(int(l / n * N_BINS), N_BINS - 1)
                grid[ti, b] = max(grid[ti, b], v)

        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")

        # Add suite dividers
        suites_seen = []
        for r in mrows:
            if r["suite"] not in suites_seen:
                suites_seen.append(r["suite"])
        suite_starts = {}
        for r in mrows:
            ti = task_order.index(r["task_label"])
            suite_starts.setdefault(r["suite"], ti)
        for s, start in suite_starts.items():
            if start > 0:
                ax.axhline(start - 0.5, color="#ff6600", linewidth=1.2,
                           linestyle="--", alpha=0.7)
                ax.text(N_BINS - 0.5, start - 0.5, s,
                        va="bottom", ha="right", fontsize=6,
                        color="#cc4400")

        ax.set_xticks(np.linspace(0, N_BINS - 1, 5))
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.set_yticks(range(n_t))
        ax.set_yticklabels(task_order, fontsize=8)
        ax.set_xlabel("Normalised layer depth", fontsize=9)
        ax.set_title(mname.split("/")[-1], fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=axes[0][-1], fraction=0.046, pad=0.04,
                 label="FFN active (1) / skip (0)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Grouped bar — compute horizon by task
# ─────────────────────────────────────────────────────────────────────────────

def plot_horizon_bars(rows: List[Dict], out_path: Path) -> None:
    models = list(dict.fromkeys(r["model"] for r in rows))
    tasks  = list(dict.fromkeys(r["task_label"] for r in rows))
    n_m    = len(models)
    width  = 0.8 / n_m
    colours = plt.cm.tab10(np.linspace(0, 0.9, n_m))

    fig, ax = plt.subplots(figsize=(max(14, len(tasks) * 1.5), 5))
    for mi, mname in enumerate(models):
        lookup = {r["task_label"]: r for r in rows if r["model"] == mname}
        xs, ys = [], []
        for ti, task in enumerate(tasks):
            if task in lookup:
                xs.append(ti + mi * width - (n_m - 1) * width / 2)
                ys.append(lookup[task]["compute_horizon_pct"])
        ax.bar(xs, ys, width=width * 0.9,
               color=colours[mi], label=mname.split("/")[-1], alpha=0.85)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=42, ha="right", fontsize=8)
    ax.set_ylabel("Compute horizon (normalised depth)", fontsize=10)
    ax.set_title("Compute horizon by task and model\n"
                 "(how deep into the network the last FFN fires)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--",
               label="50% depth")
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    # Colour x-tick labels by task type
    for tick, task in zip(ax.get_xticklabels(), tasks):
        tick.set_color(_task_colour(task))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Horizon    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Scatter — fragmentation vs late-attn fraction
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(rows: List[Dict], out_path: Path) -> None:
    markers    = ["o", "s", "^", "D", "v", "P", "*", "X"]
    model_mark = {}
    for r in rows:
        if r["model"] not in model_mark:
            model_mark[r["model"]] = markers[len(model_mark) % len(markers)]

    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(0)
    for r in rows:
        jx = rng.uniform(-0.06, 0.06)
        jy = rng.uniform(-0.012, 0.012)
        ax.scatter(
            r["fragmentation"] + jx,
            r["late_attn_frac"] + jy,
            c=_task_colour(r["task_label"]),
            marker=model_mark[r["model"]],
            s=80, alpha=0.80, linewidths=0.6, edgecolors="#444",
        )

    # Model legend
    for mname, mk in model_mark.items():
        ax.scatter([], [], marker=mk, color="#777", s=70,
                   label=mname.split("/")[-1])
    ax.legend(fontsize=8, title="Model", loc="upper left",
              framealpha=0.9)

    # Task-type colour patches
    seen_colours: Dict[str, str] = {}
    for r in rows:
        c = _task_colour(r["task_label"])
        key = r["task_label"]
        if c not in seen_colours.values():
            seen_colours[key] = c
    from matplotlib.patches import Patch
    patches = [Patch(color=c, label=lbl)
               for lbl, c in list(seen_colours.items())[:12]]
    ax.legend(handles=patches, fontsize=7, title="Task (colour)",
              loc="upper right", framealpha=0.9, ncol=2)

    ax.set_xlabel("Fragmentation  (# separate FFN-skip islands)", fontsize=10)
    ax.set_ylabel("Late-layer attention fraction  (last 3 layers)", fontsize=10)
    ax.set_title("Skip-island fragmentation vs late-layer attention\n"
                 "Marker shape = model  |  Colour = task type", fontsize=11)
    ax.set_xlim(-0.4, max(r["fragmentation"] for r in rows) + 0.8)
    ax.set_ylim(-0.08, 1.08)
    ax.axhline(0.3, color="#cccccc", linewidth=0.8, linestyle=":")
    ax.axvline(1.5, color="#cccccc", linewidth=0.8, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Line — compute horizon vs hop count (deep_chains suite)
# ─────────────────────────────────────────────────────────────────────────────

def plot_horizon_lines(rows: List[Dict], out_path: Path) -> None:
    """
    For the deep_chains suite: x = hop count, y = compute_horizon_pct.
    One line per model. Shows how fast the compute horizon retreats.
    """
    chain_rows = [r for r in rows if "hop" in r["task_label"].lower()
                  and "chain" in r["task_label"].lower()]
    if not chain_rows:
        return

    def _hop(label: str) -> int:
        import re
        m = re.search(r"(\d+)-hop", label.lower())
        return int(m.group(1)) if m else 0

    models  = list(dict.fromkeys(r["model"] for r in chain_rows))
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(models)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for mi, mname in enumerate(models):
        pts = sorted(
            [(r["task_label"], r["compute_horizon_pct"])
             for r in chain_rows if r["model"] == mname],
            key=lambda t: _hop(t[0]),
        )
        if not pts:
            continue
        xs = [_hop(lbl) for lbl, _ in pts]
        ys = [h for _, h in pts]
        ax.plot(xs, ys, "o-", color=colours[mi],
                label=mname.split("/")[-1], linewidth=2, markersize=7)

    ax.set_xlabel("k  (chain hop count)", fontsize=11)
    ax.set_ylabel("Compute horizon (normalised depth)", fontsize=11)
    ax.set_title("Compute horizon vs chain depth\n"
                 "(does the model go deeper or shallower for harder tasks?)",
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted(set(_hop(r["task_label"]) for r in chain_rows)))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Hop lines  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table (stdout + plain-text file)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(rows: List[Dict], out_path: Path) -> None:
    """
    Prints four pivot tables: k, horizon, fragmentation, late_attn.
    Each table: rows = tasks, cols = models.
    """
    models = list(dict.fromkeys(r["model"] for r in rows))
    tasks  = list(dict.fromkeys(r["task_label"] for r in rows))
    short  = [m.split("/")[-1][:18] for m in models]

    def pivot(field):
        table: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            table.setdefault(r["task_label"], {})[r["model"]] = r.get(field, "")
        return table

    lines = []
    for field, title, fmt in [
        ("k",                   "Active edges  k",                 lambda v: f"{int(v):4d}"),
        ("compute_horizon_pct", "Compute horizon  (norm. depth)",  lambda v: f"{float(v):.2f}"),
        ("fragmentation",       "Fragmentation  (# skip islands)", lambda v: f"{int(v):4d}"),
        ("late_attn_frac",      "Late-layer attention fraction",   lambda v: f"{float(v):.2f}"),
    ]:
        col_w = 20
        header = f"{'Task':<30}" + "".join(f"{s:>{col_w}}" for s in short)
        sep    = "─" * len(header)
        lines += [f"\n{'━'*len(header)}", f"  {title}", f"{'━'*len(header)}", header, sep]
        piv = pivot(field)
        for task in tasks:
            row_str = f"{task:<30}"
            for mname in models:
                val = piv.get(task, {}).get(mname, "")
                row_str += f"{fmt(val) if val != '' else '   —':>{col_w}}"
            lines.append(row_str)

    text = "\n".join(lines)
    print(text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n  Summary    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    suite_names = ", ".join(TASK_SUITES.keys())
    parser = argparse.ArgumentParser(
        description="Skip-profile analysis across models × task suites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available suites:\n  {suite_names}",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["gpt2", "gpt2-medium", "EleutherAI/pythia-160m"],
        help="Model names (TransformerLens / HuggingFace).",
    )
    parser.add_argument(
        "--suites", nargs="+",
        default=["reasoning", "deep_chains"],
        help="Task suite names.",
    )
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--mass_coverage",type=float, default=MASS_COVERAGE)
    parser.add_argument(
        "--out", default="results/skip_analysis",
        help="Output path stem (no extension).",
    )
    args = parser.parse_args()

    all_rows: List[Dict] = []
    for mname in args.models:
        rows = analyse_model(mname, args.suites, args.device,
                             args.hf_token, args.mass_coverage)
        all_rows.extend(rows)

    if not all_rows:
        print("No results produced — check model names and suite names.")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    save_csv   (all_rows, out.with_suffix(".csv"))
    print_summary(all_rows, out.parent / (out.stem + "_summary.txt"))
    plot_heatmap      (all_rows, out.parent / (out.stem + "_heatmap.png"))
    plot_horizon_bars (all_rows, out.parent / (out.stem + "_horizon.png"))
    plot_scatter      (all_rows, out.parent / (out.stem + "_scatter.png"))
    plot_horizon_lines(all_rows, out.parent / (out.stem + "_lines.png"))

    print(f"\n{'='*62}")
    print(f"Done.  All outputs in:  {out.parent}/")
    print(f"  {out.stem}.csv")
    print(f"  {out.stem}_summary.txt")
    print(f"  {out.stem}_heatmap.png")
    print(f"  {out.stem}_horizon.png")
    print(f"  {out.stem}_scatter.png")
    print(f"  {out.stem}_lines.png")


if __name__ == "__main__":
    main()
