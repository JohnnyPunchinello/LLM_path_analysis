#!/usr/bin/env python3
"""
active_subgraph_viz.py
======================
Show the active computation subgraph of a transformer for a set of tasks.
One panel per task, stacked vertically.

Layout per panel
----------------
     [A0] [A1] [A2]  ...  [A_{L-1}]      ← attention blocks (above backbone)
  ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●   ← residual stream backbone
     [M0] [M1] [M2]  ...  [M_{L-1}]      ← MLP blocks (below backbone)

  Active block  → coloured box + bold border + diagonal lines to backbone.
  Inactive block → faint grey outline only, no connection lines.

  Residual (skip) connection between layers l and l+1:
    • Bold black backbone segment when BOTH flanking stream nodes are active.
    • Thin dashed grey segment otherwise.
  A stream node is "active" if at least one adjacent compute block is active.

Usage
-----
  python active_subgraph_viz.py \\
      --model gpt2 \\
      --tasks "The cat sat on the mat." \\
              "What is the square root of 144?" \\
              "Alice is the mother of Bob. Bob is the mother of Carol. Who is Alice's grandchild?" \\
      --mass_coverage 0.9 \\
      --out active_subgraphs.png
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

import torch
import transformer_lens
from transformer_lens import HookedTransformer

from path_analyzer import PathAnalyzer, select_active_edges_by_mass_coverage


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
_ATTN_CMAP       = plt.cm.YlOrRd          # yellow → orange → red
_MLP_CMAP        = plt.cm.YlGnBu          # yellow-green → blue
_ATTN_EDGE_COL   = "#d62728"              # strong red
_MLP_EDGE_COL    = "#1f77b4"              # strong blue
_INACTIVE_FACE   = "#f4f4f4"
_INACTIVE_EDGE   = "#c8c8c8"
_BACKBONE_ON     = "#1a1a1a"              # near-black: active residual
_BACKBONE_OFF    = "#d0d0d0"              # light grey: inactive residual
_NODE_ON         = "#1a1a1a"
_NODE_OFF        = "#d0d0d0"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _is_llama_family(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("llama", "mistral", "gemma", "falcon"))


def load_model(model_name: str, device: str = "cuda",
               hf_token: Optional[str] = None) -> HookedTransformer:
    import os
    resolved_token = (hf_token
                      or os.environ.get("HF_TOKEN")
                      or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if resolved_token:
        import huggingface_hub
        huggingface_hub.login(token=resolved_token, add_to_git_credential=False)
        print("Logged in to HuggingFace Hub.")

    device = device if torch.cuda.is_available() else "cpu"
    extra: dict = {}
    if _is_llama_family(model_name):
        extra = dict(fold_ln=False, center_writing_weights=False,
                     center_unembed=False)

    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = HookedTransformer.from_pretrained(
                model_name, quantization_config=bnb,
                device_map="auto", **extra,
            )
            print(f"Loaded {model_name} with 4-bit NF4.")
            return model
        except Exception as e:
            print(f"4-bit failed ({e}); loading in full precision.")

    model = HookedTransformer.from_pretrained(model_name, **extra)
    model.eval()
    if device == "cuda":
        model = model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Attribution → active masks
# ─────────────────────────────────────────────────────────────────────────────

def get_active_subgraph(
    model: HookedTransformer,
    text: str,
    mass_coverage: float = 0.90,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, List[bool], List[bool], float, int]:
    """
    Tokenise `text`, run attribution patching, apply mass-coverage selection.

    Returns
    -------
    attn_scores, mlp_scores : np.ndarray  [n_layers]  raw AtP magnitudes
    active_attn, active_mlp : List[bool]  [n_layers]  mass-coverage mask
    epsilon                 : float       coverage threshold
    k_edges                 : int         number of active edges
    """
    analyzer = PathAnalyzer(model)

    tokens = model.to_tokens(text)
    if tokens.shape[-1] > 512:
        tokens = tokens[:, -512:]
    tokens = tokens.to(next(model.parameters()).device)

    attn_t, mlp_t = analyzer.compute_attribution_scores(tokens, target_pos=-1)
    attn_np = attn_t.cpu().float().numpy()
    mlp_np  = mlp_t.cpu().float().numpy()

    act_a, act_m, eps, k = select_active_edges_by_mass_coverage(
        attn_np, mlp_np, mass_fraction=mass_coverage
    )
    return attn_np, mlp_np, act_a, act_m, eps, k


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stream_node_active(
    l: int,
    n_layers: int,
    act_a: List[bool],
    act_m: List[bool],
) -> bool:
    """
    Stream node at boundary index l (0 = input, n_layers = output).
    Active iff at least one adjacent compute block is active.
    Input and output nodes are always active.
    """
    if l == 0 or l == n_layers:
        return True
    left  = (act_a[l - 1] or act_m[l - 1]) if l > 0         else False
    right = (act_a[l]     or act_m[l])     if l < n_layers   else False
    return left or right


def _draw_connection_lines(
    ax: plt.Axes,
    x_node_l: float, x_node_r: float, y_stream: float,
    x_box_l: float, x_box_r: float, y_box: float,
    color: str, lw: float = 1.4, alpha: float = 0.9,
) -> None:
    """
    Draw two diagonal segments forming the active-compute arc:
      stream_node_l  →  left-edge-of-box      (entry)
      right-edge-of-box  →  stream_node_r     (exit, with arrowhead)
    """
    # Entry: stream node → box left edge
    ax.plot([x_node_l, x_box_l], [y_stream, y_box],
            color=color, lw=lw, alpha=alpha, zorder=4, solid_capstyle="round")
    # Exit: box right edge → stream node (arrowhead at destination)
    ax.annotate(
        "", xy=(x_node_r, y_stream), xytext=(x_box_r, y_box),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=9, connectionstyle="arc3,rad=0.0"),
        zorder=4,
    )


def draw_panel(
    ax: plt.Axes,
    attn_scores: np.ndarray,
    mlp_scores: np.ndarray,
    active_attn: List[bool],
    active_mlp: List[bool],
    title: str,
    task_text: str,
    epsilon: float,
    k_edges: int,
    is_attn_only: bool = False,
) -> None:
    """Draw one task's active subgraph onto ax."""
    n = len(attn_scores)

    # ── Layout constants ──────────────────────────────────────────────────────
    col_w    = 1.0       # x-distance between adjacent stream nodes
    bw       = 0.62      # box width
    bh       = 0.46      # box height
    y_attn   = 1.05      # y-centre of attention boxes
    y_mlp    = -1.05     # y-centre of MLP boxes
    y_stream = 0.0       # residual backbone y
    fs_label = max(5, 8 - n // 10)   # layer-label font size, shrinks for wide models

    # Normalise scores to [0, 1] for colour intensity
    all_sc = np.concatenate([attn_scores, mlp_scores])
    s_max  = all_sc.max() if all_sc.max() > 0.0 else 1.0

    # ── Stream-node activity ──────────────────────────────────────────────────
    node_act = [_stream_node_active(l, n, active_attn, active_mlp)
                for l in range(n + 1)]

    # ── Residual backbone ─────────────────────────────────────────────────────
    for l in range(n):
        x0, x1 = l * col_w, (l + 1) * col_w
        if node_act[l] and node_act[l + 1]:
            ax.plot([x0, x1], [y_stream, y_stream],
                    color=_BACKBONE_ON, lw=2.5, zorder=2, solid_capstyle="butt")
        else:
            ax.plot([x0, x1], [y_stream, y_stream],
                    color=_BACKBONE_OFF, lw=0.9, zorder=2, linestyle="--")

    # ── Stream nodes (circles on backbone) ────────────────────────────────────
    for l in range(n + 1):
        c  = _NODE_ON  if node_act[l] else _NODE_OFF
        ec = "white"   if node_act[l] else _NODE_OFF
        ax.scatter([l * col_w], [y_stream], s=60, c=c, zorder=6,
                   linewidths=1.5, edgecolors=ec)

    # "in" / "out" labels
    ax.text(-0.15, y_stream, "in",  ha="right", va="center",
            fontsize=7, color="#555555", style="italic")
    ax.text(n * col_w + 0.15, y_stream, "out", ha="left", va="center",
            fontsize=7, color="#555555", style="italic")

    # ── Compute blocks ────────────────────────────────────────────────────────
    for l in range(n):
        x_c  = l * col_w + col_w / 2     # box x-centre
        x_nl = l * col_w                  # left stream node
        x_nr = (l + 1) * col_w            # right stream node

        a_norm = float(attn_scores[l]) / s_max
        m_norm = float(mlp_scores[l])  / s_max

        # ─ Attention ─────────────────────────────────────────────────────────
        if active_attn[l]:
            fc_a = _ATTN_CMAP(0.30 + 0.70 * a_norm)
            ec_a, lw_a = _ATTN_EDGE_COL, 2.2
            tc_a, fw_a = "black", "bold"
        else:
            fc_a = _INACTIVE_FACE
            ec_a, lw_a = _INACTIVE_EDGE, 0.6
            tc_a, fw_a = "#aaaaaa", "normal"

        ax.add_patch(FancyBboxPatch(
            (x_c - bw / 2, y_attn - bh / 2), bw, bh,
            boxstyle="round,pad=0.03",
            facecolor=fc_a, edgecolor=ec_a, linewidth=lw_a, zorder=3,
        ))
        ax.text(x_c, y_attn, f"A{l}", ha="center", va="center",
                fontsize=fs_label, color=tc_a, fontweight=fw_a)

        if active_attn[l]:
            _draw_connection_lines(
                ax, x_nl, x_nr, y_stream,
                x_c - bw / 2, x_c + bw / 2, y_attn,
                _ATTN_EDGE_COL,
            )

        # ─ MLP ───────────────────────────────────────────────────────────────
        if not is_attn_only:
            if active_mlp[l]:
                fc_m = _MLP_CMAP(0.30 + 0.70 * m_norm)
                ec_m, lw_m = _MLP_EDGE_COL, 2.2
                tc_m, fw_m = "black", "bold"
            else:
                fc_m = _INACTIVE_FACE
                ec_m, lw_m = _INACTIVE_EDGE, 0.6
                tc_m, fw_m = "#aaaaaa", "normal"

            ax.add_patch(FancyBboxPatch(
                (x_c - bw / 2, y_mlp - bh / 2), bw, bh,
                boxstyle="round,pad=0.03",
                facecolor=fc_m, edgecolor=ec_m, linewidth=lw_m, zorder=3,
            ))
            ax.text(x_c, y_mlp, f"M{l}", ha="center", va="center",
                    fontsize=fs_label, color=tc_m, fontweight=fw_m)

            if active_mlp[l]:
                _draw_connection_lines(
                    ax, x_nl, x_nr, y_stream,
                    x_c - bw / 2, x_c + bw / 2, y_mlp,
                    _MLP_EDGE_COL,
                )

    # ── Panel title & stats ───────────────────────────────────────────────────
    n_a = sum(active_attn)
    n_m = sum(active_mlp)
    stats = (f"k = {k_edges} active edges  |  "
             f"attn: {n_a}/{n}  MLP: {n_m}/{n}  |  ε = {epsilon:.4f}")

    wrapped = textwrap.shorten(task_text, width=90, placeholder=" ...")
    ax.set_title(f'{title}\n\u201c{wrapped}\u201d',
                 fontsize=8.5, fontweight="bold", pad=5, loc="left")
    ax.text(
        n * col_w / 2, y_mlp - 0.72, stats,
        ha="center", va="top", fontsize=7.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fafafa",
                  edgecolor="#cccccc", alpha=0.95),
    )

    ax.set_xlim(-0.6, n * col_w + 0.6)
    ax.set_ylim(y_mlp - 1.0, y_attn + 0.70)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────

def _add_legend(fig: plt.Figure) -> None:
    handles = [
        mpatches.Patch(facecolor=_ATTN_CMAP(0.75), edgecolor=_ATTN_EDGE_COL,
                       linewidth=1.5, label="Active attention block"),
        mpatches.Patch(facecolor=_MLP_CMAP(0.75),  edgecolor=_MLP_EDGE_COL,
                       linewidth=1.5, label="Active MLP block"),
        mpatches.Patch(facecolor=_INACTIVE_FACE,    edgecolor=_INACTIVE_EDGE,
                       linewidth=0.6, label="Inactive block"),
        Line2D([0], [0], color=_BACKBONE_ON, lw=2.5,
               label="Active residual connection"),
        Line2D([0], [0], color=_BACKBONE_OFF, lw=0.9, linestyle="--",
               label="Inactive residual connection"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise active computation subgraphs for a set of tasks."
    )
    parser.add_argument(
        "--model", default="gpt2",
        help="TransformerLens model name (default: gpt2). "
             "Use NousResearch/Meta-Llama-3-8B for Llama-3.",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=[
            "The cat sat on the mat.",
            "What is the square root of 144? The answer is",
            "Alice is the mother of Bob. Bob is the mother of Carol. "
            "Who is Alice's grandchild? The answer is",
            "Premise: All mammals breathe air. Whales are mammals. "
            "Conclusion: Whales breathe air. This is",
        ],
        help="List of task prompts to visualise.",
    )
    parser.add_argument(
        "--task_labels", nargs="+", default=None,
        help="Short labels for each task (default: 'Task 1', 'Task 2', ...).",
    )
    parser.add_argument(
        "--mass_coverage", type=float, default=0.90,
        help="Mass-coverage fraction for active-edge selection (default: 0.90).",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device: cuda or cpu (default: cuda; falls back to cpu automatically).",
    )
    parser.add_argument(
        "--hf_token", default=None,
        help="HuggingFace token for gated repos (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--out", default="active_subgraphs.png",
        help="Output image path (default: active_subgraphs.png).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output DPI (default: 150).",
    )
    args = parser.parse_args()

    tasks  = args.tasks
    labels = args.task_labels or [f"Task {i+1}" for i in range(len(tasks))]
    if len(labels) < len(tasks):
        labels += [f"Task {i+1}" for i in range(len(labels), len(tasks))]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = load_model(args.model, device=args.device, hf_token=args.hf_token)
    model.eval()

    n_layers   = model.cfg.n_layers
    is_ao      = bool(getattr(model.cfg, "attn_only",        False))
    is_par     = bool(getattr(model.cfg, "parallel_attn_mlp", False))

    print(f"  n_layers={n_layers}  attn_only={is_ao}  parallel={is_par}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_tasks    = len(tasks)
    panel_h    = 3.8      # height per panel (inches)
    fig_w      = max(12, n_layers * 0.55)
    fig_h      = panel_h * n_tasks + 0.7   # 0.7 for legend

    fig, axes = plt.subplots(
        n_tasks, 1,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )
    if n_tasks == 1:
        axes = [axes]

    fig.suptitle(
        f"Active Computation Subgraphs  –  {args.model}  "
        f"(mass coverage = {args.mass_coverage:.0%})",
        fontsize=10, fontweight="bold", y=1.0,
    )

    # ── Per-task processing ───────────────────────────────────────────────────
    for i, (ax, text, label) in enumerate(zip(axes, tasks, labels)):
        print(f"\nProcessing: {label!r}")
        print(f"  Prompt: {text[:80]}{'…' if len(text)>80 else ''}")

        try:
            attn_sc, mlp_sc, act_a, act_m, eps, k = get_active_subgraph(
                model, text,
                mass_coverage=args.mass_coverage,
                device=args.device,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            ax.text(0.5, 0.5, f"[Error: {e}]", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")
            ax.axis("off")
            continue

        n_a = sum(act_a)
        n_m = sum(act_m)
        print(f"  Active edges: k={k}  attn={n_a}/{n_layers}  "
              f"mlp={n_m}/{n_layers}  ε={eps:.5f}")

        draw_panel(
            ax,
            attn_scores=attn_sc,
            mlp_scores=mlp_sc,
            active_attn=act_a,
            active_mlp=act_m,
            title=label,
            task_text=text,
            epsilon=eps,
            k_edges=k,
            is_attn_only=is_ao,
        )

    _add_legend(fig)

    plt.subplots_adjust(
        left=0.02, right=0.98,
        top=0.97,  bottom=0.06,
        hspace=0.55,
    )

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved → {out_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
