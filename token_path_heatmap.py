#!/usr/bin/env python3
"""
token_path_heatmap.py
=====================
Token-Wise Path Recruitment Heatmap for Path Distribution Theory.

For each token position t in a prompt the script:
  1. Runs a backward pass targeting logit[t] → predicted_token
     (attribution anchored at blocks.0.hook_resid_pre to avoid backprop
     through quantised embedding weights)
  2. Computes position-specific AtP scores:
       score_l = | grad_{attn_out_l}[t] · attn_out_l[t] |.mean(d_model)
  3. Thresholds at the 60th percentile → top-40% active subgraph G_t
  4. Runs Algorithm 1 (path DP) on G_t to get empirical E[L]_t

Hypothesis tested
-----------------
  Long residual paths are recruited specifically for COMPLEX reasoning tokens.
  Path recruitment should increase monotonically with task difficulty:

    grammar  <  simple_math  <  hard_math  <  logic_puzzle

  Metrics:
    Recruitment Delta(X) = max_E[L](X) − max_E[L](grammar)
    A positive and increasing delta across difficulty tiers confirms the
    Complexity-Matching prediction of Path Distribution Theory.

Prompt difficulty tiers (four prompts run by default)
------------------------------------------------------
  grammar     — pure syntactic completion, no arithmetic
  simple_math — single-step multiplication + subtraction
  hard_math   — 3-digit multiplication with explicit partial products
                (two carry steps: units digit, tens digit, sum)
  logic_puzzle— multi-step constraint deduction (5-person ordering)

Active subgraph
---------------
  active_fraction = 0.40  →  top 40 % highest-attribution edges kept
  epsilon          = 60th percentile of all 2L AtP scores per position

Outputs
-------
  results/token_heatmap.png              — 4-row colour-coded token strips
  results/token_heatmap_timeseries.png   — E[L] line plot, all 4 prompts
  results/token_heatmap.json             — full data + per-prompt deltas

Usage
-----
  # Default: Pythia-2.8b, GPU, all 4 prompts
  python token_path_heatmap.py

  # Smaller model (CPU-feasible)
  python token_path_heatmap.py --model EleutherAI/pythia-1b --device cpu

  # Override individual prompts
  python token_path_heatmap.py \\
      --grammar   "The sun rises in the east and sets in the" \\
      --hard_math "What is 312 * 45? 312*5=1560, 312*40=12480, total:"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import torch

try:
    import seaborn as sns
    sns.set_theme(style="white", font_scale=1.05)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Default test prompts ──────────────────────────────────────────────────────
# Each prompt is a prefix whose per-token E[L] will be measured.
# They are ordered by expected computational depth so the heatmap rows
# display a clear difficulty gradient.

PROMPTS = {
    # Tier 0 — pure grammar, no arithmetic, no deduction
    "grammar": (
        "The quick brown fox jumps over the lazy dog and then it"
    ),

    # Tier 1 — single-step reasoning: one multiplication, one subtraction
    "simple_math": (
        "Question: If I have 3 sets of 4 apples and I eat 2, "
        "how many are left? "
        "Answer: 3 * 4 = 12 apples. 12 - 2 ="
    ),

    # Tier 2 — 3-digit × 2-digit multiplication via explicit partial products.
    # The model must track two partial results and carry them into a final sum.
    # Key computation tokens: "1488", "7440", "8928", each "=".
    "hard_math": (
        "Question: A factory produces 248 widgets per hour and runs for "
        "36 hours. How many widgets total? "
        "Answer: 248 * 6 = 1488. 248 * 30 = 7440. 1488 + 7440 ="
    ),

    # Tier 3 — multi-step constraint satisfaction.
    # Five people, four clues, each deduction narrows the solution space.
    # Key reasoning tokens: "=5", "=4", "=3", "=2", "Bob=".
    "logic_puzzle": (
        "Puzzle: Five friends (Alice, Bob, Carol, Dave, Eve) stand in a "
        "numbered line 1–5. Dave is last. Eve is directly in front of Dave. "
        "Alice is not first. Bob is directly behind Carol. "
        "Solution: Dave=5, Eve=4, Alice=3. Remaining spots 1,2 with Bob "
        "directly behind Carol, so Carol=1, Bob="
    ),
}

# Ordered from easiest to hardest for consistent plot layout
PROMPT_ORDER = ["grammar", "simple_math", "hard_math", "logic_puzzle"]

# Top-N% active subgraph threshold  (raised from 0.20 to expose richer contrast)
ACTIVE_FRACTION  = 0.40          # keep top 40 % highest-attribution edges
EPSILON_QUANTILE = 1.0 - ACTIVE_FRACTION   # → 60th percentile threshold


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_name: str, device: str = "cuda"):
    from transformer_lens import HookedTransformer

    log.info("Loading %s …", model_name)
    try:
        import bitsandbytes  # noqa: F401
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0,
                                 llm_int8_has_fp16_weight=False)
        hf_m = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16)
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hf_m, dtype=torch.float16, move_to_device=False)
        model.eval()
        log.info("  -> 8-bit quantised load OK")
        return model
    except Exception as e:
        log.debug("  8-bit failed (%s) — fp16 fallback.", e)

    model = HookedTransformer.from_pretrained(
        model_name, dtype=torch.float16, device=device)
    model.eval()
    log.info("  -> fp16 load OK")
    return model


# =============================================================================
# Position-specific Attribution Patching
# =============================================================================

def _position_atp_scores(
    model,
    tokens:     torch.Tensor,
    position:   int,
    n_layers:   int,
    is_attn_only: bool,
    device:     str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute per-layer AtP scores anchored at sequence position `position`.

    For each layer l:
        attn_score[l] = | grad_{attn_out_l}[position] · attn_out_l[position] |
                          .mean(dim=d_model)

    A single backward pass is used (anchored at blocks.0.hook_resid_pre to
    avoid gradient flow through quantised / frozen embedding weights).

    Parameters
    ----------
    tokens    : LongTensor [1, seq_len]
    position  : sequence position for which to compute attribution
    n_layers  : model depth
    is_attn_only : True for attention-only models

    Returns
    -------
    attn_scores : np.ndarray [n_layers]
    mlp_scores  : np.ndarray [n_layers]
    pred_token  : int — argmax token predicted at this position
    """
    # ── get predicted token (no grad needed) ────────────────────────────────
    with torch.no_grad():
        logits_det = model(tokens, return_type="logits")
        pred_token = int(logits_det[0, position].argmax())

    # ── set up differentiable forward pass ──────────────────────────────────
    # attn_acts / mlp_acts store the live tensors (with grad_fn after anchor)
    attn_acts: List[Optional[torch.Tensor]] = [None] * n_layers
    mlp_acts:  List[Optional[torch.Tensor]] = [None] * n_layers
    fwd_hooks: list = []

    # Anchor: detach at layer-0 residual input, re-attach leaf grad.
    # This prevents backprop through (quantised) embedding weights while
    # keeping the graph live for all transformer blocks.
    def _anchor(act, hook):
        return act.detach().float().requires_grad_(True)
    fwd_hooks.append(("blocks.0.hook_resid_pre", _anchor))

    for layer in range(n_layers):
        def _h_attn(act, hook, l=layer):
            if act.requires_grad:
                act.retain_grad()
            attn_acts[l] = act
            return act
        fwd_hooks.append((f"blocks.{layer}.hook_attn_out", _h_attn))

        if not is_attn_only:
            def _h_mlp(act, hook, l=layer):
                if act.requires_grad:
                    act.retain_grad()
                mlp_acts[l] = act
                return act
            fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", _h_mlp))

    try:
        with torch.enable_grad():
            logits = model.run_with_hooks(
                tokens, fwd_hooks=fwd_hooks, return_type="logits")
            logits[0, position, pred_token].backward()
    except RuntimeError as exc:
        warnings.warn(f"Gradient computation at position {position} failed: {exc}")
        return np.zeros(n_layers), np.zeros(n_layers), pred_token

    # ── position-specific gradient × activation ──────────────────────────────
    attn_scores = np.zeros(n_layers)
    mlp_scores  = np.zeros(n_layers)

    for l in range(n_layers):
        a = attn_acts[l]
        if a is not None and a.grad is not None:
            # Shape of a.grad: [1, seq_len, d_model]
            # We want attribution at this specific sequence position
            act_l  = a.detach()[0, position].float()   # [d_model]
            grad_l = a.grad[0, position].float()        # [d_model]
            attn_scores[l] = float((grad_l * act_l).abs().mean())

        m = mlp_acts[l]
        if m is not None and m.grad is not None:
            act_l  = m.detach()[0, position].float()
            grad_l = m.grad[0, position].float()
            mlp_scores[l] = float((grad_l * act_l).abs().mean())

    return attn_scores, mlp_scores, pred_token


# =============================================================================
# Main per-token analysis loop
# =============================================================================

def analyse_prompt(
    model,
    analyzer,          # PathAnalyzer instance
    prompt_text: str,
    label:       str,
    device:      str,
    active_fraction: float = ACTIVE_FRACTION,
    skip_bos:    bool = True,
) -> Dict[str, Any]:
    """
    Run the full per-token path-length analysis for one prompt.

    Returns a dict with token texts, predicted tokens, E[L] values,
    and attribution score arrays for downstream inspection.
    """
    from path_analyzer import _to_metrics   # module-level helper

    epsilon_q = 1.0 - active_fraction       # 80th percentile → top 20 % active

    # Tokenise
    tokens = model.to_tokens(prompt_text, prepend_bos=True).to(device)
    seq_len = tokens.shape[-1]
    log.info("[%s] %d tokens", label, seq_len)

    # Token display strings (decoded one-by-one for clean labels)
    token_strs: List[str] = []
    for idx in tokens[0].tolist():
        raw = model.tokenizer.decode([idx])
        token_strs.append(raw)

    # Positions to analyse.
    # Position t → predicts what comes after position t.
    # We skip position 0 (BOS) as it has no prior context.
    start_pos = 1 if skip_bos else 0
    positions = list(range(start_pos, seq_len))

    el_list:         List[float]      = []
    pred_tokens:     List[str]        = []
    all_attn_scores: List[np.ndarray] = []
    all_mlp_scores:  List[np.ndarray] = []
    active_edge_counts: List[int]     = []

    n_layers    = analyzer.n_layers
    is_attn_only = analyzer.is_attn_only

    for i, t in enumerate(positions):
        log.info("  [%s] position %d/%d — token '%s'",
                 label, i + 1, len(positions), repr(token_strs[t]))

        a_sc, m_sc, pred_tok = _position_atp_scores(
            model, tokens, t, n_layers, is_attn_only, device)

        # Threshold: keep top (active_fraction * 100)% edges
        all_sc = np.concatenate([a_sc, m_sc])
        epsilon = float(np.quantile(all_sc, epsilon_q))

        active_attn = (a_sc > epsilon).tolist()
        active_mlp  = (m_sc > epsilon).tolist()
        n_active    = sum(active_attn) + sum(active_mlp)

        counts  = analyzer._path_count_dp(active_attn, active_mlp)
        metrics = _to_metrics(counts)

        el_list.append(metrics.mean_path_length)
        pred_tokens.append(model.tokenizer.decode([pred_tok]))
        all_attn_scores.append(a_sc)
        all_mlp_scores.append(m_sc)
        active_edge_counts.append(n_active)

        log.info("    E[L]=%.2f  active_edges=%d  ε=%.2e",
                 metrics.mean_path_length, n_active, epsilon)

    # Pad BOS position with NaN so index aligns with token_strs
    if skip_bos:
        el_padded   = [float("nan")] + el_list
        pred_padded = ["—"]          + pred_tokens
        n_act_padded= [0]            + active_edge_counts
    else:
        el_padded   = el_list
        pred_padded = pred_tokens
        n_act_padded= active_edge_counts

    return {
        "label":          label,
        "prompt":         prompt_text,
        "token_strs":     token_strs,
        "el_values":      el_padded,
        "pred_tokens":    pred_padded,
        "n_active_edges": n_act_padded,
        "max_el":         float(np.nanmax(el_padded)),
        "mean_el":        float(np.nanmean(el_padded)),
        "min_el":         float(np.nanmin(el_padded)),
        # Top-recruited positions (excluding BOS)
        "top5_positions": sorted(
            [(i, el) for i, el in enumerate(el_padded)
             if not np.isnan(el)],
            key=lambda x: x[1], reverse=True
        )[:5],
    }


# =============================================================================
# Visualisation — recruitment heatmap
# =============================================================================

_CLEAN_SUBS = str.maketrans({
    "\u0120": " ",   # Ġ  (GPT-2 space prefix)
    "\u2581": " ",   # ▁  (SentencePiece space prefix)
    "\n":     "↵",
    "\t":     "→",
})

def _clean_tok(s: str, maxlen: int = 7) -> str:
    s = s.translate(_CLEAN_SUBS).strip()
    if not s:
        return "·"
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def _brightness(rgba) -> float:
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def draw_token_strip(
    ax: plt.Axes,
    token_strs:  List[str],
    el_values:   List[float],
    vmin:        float,
    vmax:        float,
    cmap,
    norm,
    label_title: str,
    ana_mean_el: float,
    annotate_top: int = 3,
) -> None:
    """
    Draw a single horizontal strip of coloured token boxes.

    Each token box is 1 unit wide × 1 unit tall.
    Colour = E[L]_t on the shared coolwarm/YlOrRd scale.
    """
    n = len(token_strs)

    # Sort indices by E[L] for top-token annotation
    valid_el = [(i, v) for i, v in enumerate(el_values) if not np.isnan(v)]
    top_idx  = {i for i, _ in sorted(valid_el, key=lambda x: x[1], reverse=True)[:annotate_top]}

    for i, (tok, el) in enumerate(zip(token_strs, el_values)):
        is_nan = np.isnan(el)
        color  = "#cccccc" if is_nan else cmap(norm(el))

        # Box
        rect = mpatches.FancyBboxPatch(
            (i, 0), 1, 1,
            boxstyle="square,pad=0.0",
            facecolor=color,
            edgecolor="white",
            linewidth=1.8,
            zorder=1,
        )
        ax.add_patch(rect)

        # Token label (top half of box)
        tok_disp   = _clean_tok(tok)
        text_color = "black" if (is_nan or _brightness(color) > 0.45) else "white"
        ax.text(i + 0.5, 0.67, tok_disp,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold",
                color=text_color, zorder=3, clip_on=True)

        # E[L] value (lower third of box)
        if not is_nan:
            ax.text(i + 0.5, 0.25, f"{el:.1f}",
                    ha="center", va="center",
                    fontsize=6.5, color=text_color, zorder=3)

        # Crown marker on top-N tokens (plain ASCII to avoid font issues)
        if i in top_idx:
            ax.annotate("^",
                        xy=(i + 0.5, 1.0),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=11, fontweight="bold",
                        color="#cc0000", annotation_clip=False, zorder=4)

    # Analytical mean line
    ax.axhline(ana_mean_el / vmax, color="#1a1aff", lw=1.5, ls="--",
               alpha=0.7, label=f"Ana. E[L] = {ana_mean_el:.1f}")

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label_title, fontsize=11, fontweight="bold", pad=14)


def plot_heatmap(
    results:     Dict[str, Dict],
    ana_metrics,                    # PathMetrics from analytical_path_distribution()
    output_path: str,
) -> None:
    """
    Produce the two-row recruitment heatmap figure.
    Row 1 = Grammar prompt
    Row 2 = Reasoning prompt
    """
    labels  = list(results.keys())
    n_rows  = len(labels)
    max_len = max(len(r["token_strs"]) for r in results.values())

    # Global E[L] range across both prompts (shared colour scale)
    all_el  = [v for r in results.values()
                 for v in r["el_values"] if not np.isnan(v)]
    vmin    = 0.0
    vmax    = max(all_el) * 1.05 if all_el else ana_metrics.mean_path_length

    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig_w = max(max_len * 0.72, 10)
    fig_h = n_rows * 2.4 + 1.8
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h),
                              constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        r          = results[label]
        tok_strs   = r["token_strs"]
        el_vals    = r["el_values"]
        max_el_tok = tok_strs[r["el_values"].index(max(
                        v for v in el_vals if not np.isnan(v)))]
        subtitle   = (f"{label.title()}  |  "
                      f"max E[L]={r['max_el']:.2f}  "
                      f"mean E[L]={r['mean_el']:.2f}  "
                      f"(peak token: \"{_clean_tok(max_el_tok)}\")")
        draw_token_strip(ax, tok_strs, el_vals, vmin, vmax, cmap, norm,
                         label_title=subtitle,
                         ana_mean_el=ana_metrics.mean_path_length)

    # ── Colourbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.012, pad=0.01, shrink=0.85)
    cbar.set_label("Empirical Mean Path Length  E[L]_t", fontsize=10)
    # Mark analytical E[L]
    cbar.ax.axhline(ana_metrics.mean_path_length, color="#1a1aff",
                    lw=2, ls="--", alpha=0.8)
    cbar.ax.text(1.8, ana_metrics.mean_path_length,
                 f"← H_ana E[L]={ana_metrics.mean_path_length:.1f}",
                 va="center", fontsize=7.5, color="#1a1aff",
                 transform=cbar.ax.get_yaxis_transform())

    fig.suptitle(
        "Token-Wise Path Recruitment Heatmap\n"
        "^ = top-3 highest path-recruitment tokens per prompt",
        fontsize=12, fontweight="bold"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Heatmap saved → %s", output_path)
    plt.close(fig)


# =============================================================================
# Secondary figure: E[L] line plot over token position
# =============================================================================

def plot_el_timeseries(
    results:     Dict[str, Dict],
    ana_metrics,
    output_path: str,
) -> None:
    """
    Line plot of E[L]_t vs token index for all prompts on a shared axis.
    Ordered from easiest to hardest so the difficulty gradient is visible.
    Shaded bands highlight the min–max range around each curve.
    """
    # Colour palette ordered by difficulty
    PALETTE = {
        "grammar":      "#2196F3",   # blue
        "simple_math":  "#4CAF50",   # green
        "hard_math":    "#FF9800",   # orange
        "logic_puzzle": "#E53935",   # red
    }
    LINESTYLES = {
        "grammar":      (0, ()),          # solid
        "simple_math":  (0, (5, 2)),      # dashed
        "hard_math":    (0, (3, 1, 1, 1)),# dash-dot
        "logic_puzzle": (0, (1, 1)),      # dotted
    }

    # Ordered display
    ordered = [k for k in PROMPT_ORDER if k in results]
    ordered += [k for k in results if k not in ordered]

    fig, ax = plt.subplots(figsize=(14, 5.0), constrained_layout=True)

    for label in ordered:
        r    = results[label]
        toks = r["token_strs"]
        el   = r["el_values"]
        xs   = np.arange(len(toks))
        mask = [not np.isnan(v) for v in el]
        xs_v = [x for x, m in zip(xs, mask) if m]
        ys_v = [v for v, m in zip(el, mask) if m]

        color   = PALETTE.get(label, "gray")
        ls      = LINESTYLES.get(label, (0, ()))
        grammar_max = results.get("grammar", {}).get("max_el", 0.0)
        delta   = r["max_el"] - grammar_max
        dlabel  = f"  (delta={delta:+.2f})" if label != "grammar" else "  (baseline)"

        ax.plot(xs_v, ys_v, color=color, lw=2.2, dashes=ls[1],
                marker="o", ms=4.5, zorder=3,
                label=f"{label.replace('_', ' ').title()}"
                      f"  max={r['max_el']:.2f}{dlabel}")

        # Annotate peak token
        if xs_v:
            peak_i = int(np.nanargmax(el))
            ax.annotate(
                f"  '{_clean_tok(toks[peak_i])}'",
                xy=(peak_i, el[peak_i]),
                fontsize=8, color=color, va="bottom",
                xytext=(3, 3), textcoords="offset points")

    # Analytical E[L] ceiling
    ax.axhline(ana_metrics.mean_path_length, color="#1a1aff",
               ls="--", lw=1.5, alpha=0.65,
               label=f"Analytical E[L] = {ana_metrics.mean_path_length:.1f}")

    # Complexity band labels on right margin
    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Empirical  E[L]_t", fontsize=11)
    ax.set_title(
        "Path Recruitment per Token  |  "
        f"active_fraction={ACTIVE_FRACTION:.0%}  (top-40% subgraph)",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0, top=ana_metrics.mean_path_length * 1.08)
    ax.grid(True, alpha=0.35)

    ts_path = output_path.replace(".png", "_timeseries.png")
    fig.savefig(ts_path, dpi=150, bbox_inches="tight")
    log.info("Time-series saved → %s", ts_path)
    plt.close(fig)


# =============================================================================
# Output summary + JSON
# =============================================================================

def print_summary(results: Dict[str, Dict], ana_metrics, active_fraction: float) -> None:
    print()
    print("=" * 72)
    print("  TOKEN-WISE PATH RECRUITMENT SUMMARY")
    print("=" * 72)
    print(f"  Analytical E[L]  : {ana_metrics.mean_path_length:.4f}")
    print(f"  Active subgraph  : top {int(active_fraction*100)}% edges  "
          f"(epsilon = {int((1-active_fraction)*100)}th pct of AtP scores)")
    print()

    # Per-prompt table
    col_w = 14
    header = (f"  {'Prompt':<20}  {'Tokens':>6}  {'min E[L]':>8}  "
              f"{'mean E[L]':>9}  {'max E[L]':>8}  {'Delta vs grammar':>16}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    grammar_max = results.get("grammar", {}).get("max_el", 0.0)

    ordered = [k for k in PROMPT_ORDER if k in results]
    ordered += [k for k in results if k not in ordered]   # any extra prompts

    for label in ordered:
        r     = results[label]
        delta = r["max_el"] - grammar_max
        delta_str = f"{delta:+.3f}" if label != "grammar" else "  baseline"
        print(f"  {label:<20}  {len(r['token_strs']):>6}  "
              f"{r['min_el']:>8.3f}  {r['mean_el']:>9.3f}  "
              f"{r['max_el']:>8.3f}  {delta_str:>16}")
    print()

    # Per-prompt top-5 detail
    for label in ordered:
        r = results[label]
        print(f"  [{label.upper()}]  max E[L]={r['max_el']:.3f}  "
              f"mean={r['mean_el']:.3f}")
        print(f"    Top-5 recruited tokens:")
        for rank, (pos, el) in enumerate(r["top5_positions"], 1):
            tok       = _clean_tok(r["token_strs"][pos])
            predicted = _clean_tok(r["pred_tokens"][pos])
            print(f"      {rank}.  pos={pos:3d}  E[L]={el:.3f}  "
                  f"token='{tok}'  predicts='{predicted}'")
        print()

    # Recruitment Delta table
    print(f"  RECRUITMENT DELTAS  (max E[L] vs grammar baseline = {grammar_max:.3f})")
    print(f"  {'Prompt':<20}  {'Delta':>8}  Interpretation")
    print("  " + "-" * 60)
    for label in ordered:
        if label == "grammar":
            continue
        delta = results[label]["max_el"] - grammar_max
        flag  = ">>>" if delta > 2.0 else ">  " if delta > 0 else "=  "
        print(f"  {label:<20}  {delta:>+8.3f}  {flag} "
              + ("longer paths recruited" if delta > 0 else "no clear difference"))
    print()


def save_json(
    results:     Dict[str, Dict],
    ana_metrics,
    model_name:  str,
    active_fraction: float,
    output_path: str,
) -> None:
    payload = {
        "model": model_name,
        "active_subgraph_top_fraction": active_fraction,
        "analytical_mean_path_length":  ana_metrics.mean_path_length,
        "analytical_entropy_bits":      ana_metrics.entropy,
        "analytical_max_path_len":      ana_metrics.max_path_len,
        "prompts": {},
    }

    for label, r in results.items():
        # Mask NaN to null for JSON serialisability
        el_clean = [None if np.isnan(v) else round(v, 4) for v in r["el_values"]]
        payload["prompts"][label] = {
            "text":                   r["prompt"],
            "tokens":                 r["token_strs"],
            "empirical_mean_path_lengths": el_clean,
            "predicted_tokens":       r["pred_tokens"],
            "n_active_edges":         r["n_active_edges"],
            "max_E_L":                round(r["max_el"], 4),
            "mean_E_L":               round(r["mean_el"], 4),
            "min_E_L":                round(r["min_el"], 4),
            "top5_positions": [
                {"position": pos,
                 "token":    r["token_strs"][pos],
                 "E_L":      round(el, 4)}
                for pos, el in r["top5_positions"]
            ],
        }

    grammar_max = results.get("grammar", {}).get("max_el", 0.0)
    payload["recruitment_deltas_vs_grammar"] = {
        label: round(r["max_el"] - grammar_max, 4)
        for label, r in results.items()
        if label != "grammar"
    }
    # Keep legacy key for backward compatibility with plot_synergy_gap.py
    if "reasoning" in results:
        payload["recruitment_delta"] = payload["recruitment_deltas_vs_grammar"]["reasoning"]
    elif len(payload["recruitment_deltas_vs_grammar"]) > 0:
        last = list(payload["recruitment_deltas_vs_grammar"].values())[-1]
        payload["recruitment_delta"] = last

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("JSON saved → %s", output_path)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model",   default="EleutherAI/pythia-2.8b",
                        help="HuggingFace model ID (default: pythia-2.8b)")
    parser.add_argument("--device",  default="cuda",
                        help="Device: cuda or cpu (default: cuda)")

    # Per-tier prompt overrides (optional — defaults from PROMPTS dict)
    parser.add_argument("--grammar",      default=None,
                        help="Override Tier-0 grammar prompt")
    parser.add_argument("--simple_math",  default=None,
                        help="Override Tier-1 simple math prompt")
    parser.add_argument("--hard_math",    default=None,
                        help="Override Tier-2 hard math prompt (3-digit mult.)")
    parser.add_argument("--logic_puzzle", default=None,
                        help="Override Tier-3 logic puzzle prompt")

    parser.add_argument("--active_fraction", type=float, default=ACTIVE_FRACTION,
                        help=("Fraction of top-attributed edges to keep "
                              f"(default: {ACTIVE_FRACTION})"))
    parser.add_argument("--output_dir",   default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--skip_tiers",   default="",
                        help="Comma-separated tier labels to skip, e.g. logic_puzzle")
    args = parser.parse_args()

    # ── Build prompt dict ──────────────────────────────────────────────────────
    prompts: Dict[str, str] = {}
    overrides = {
        "grammar":      args.grammar,
        "simple_math":  args.simple_math,
        "hard_math":    args.hard_math,
        "logic_puzzle": args.logic_puzzle,
    }
    skip = {s.strip() for s in args.skip_tiers.split(",") if s.strip()}
    for tier in PROMPT_ORDER:
        if tier in skip:
            log.info("Skipping tier: %s", tier)
            continue
        prompts[tier] = overrides[tier] if overrides[tier] else PROMPTS[tier]

    # ── Load model ─────────────────────────────────────────────────────────────
    model = load_model(args.model, device=args.device)

    # ── PathAnalyzer ───────────────────────────────────────────────────────────
    from path_analyzer import PathAnalyzer
    analyzer    = PathAnalyzer(model)
    arch_info   = analyzer.architecture_summary()
    ana_metrics = analyzer.analytical_path_distribution()

    log.info("Architecture   : %s", arch_info["architecture"])
    log.info("n_layers       : %d",  arch_info["n_layers"])
    log.info("max_path_len   : %d",  ana_metrics.max_path_len)
    log.info("Ana. H         : %.4f bits", ana_metrics.entropy)
    log.info("Ana. E[L]      : %.4f",      ana_metrics.mean_path_length)
    log.info("active_fraction: %.0f%%",    args.active_fraction * 100)
    log.info("Running %d prompt tiers: %s", len(prompts), list(prompts.keys()))

    # ── Analyse all tiers ──────────────────────────────────────────────────────
    all_results: Dict[str, Dict] = {}
    for label, text in prompts.items():
        log.info("")
        log.info("━━━  Tier: %s  ━━━", label.upper())
        log.info("  Prompt: %s…", text[:80])
        all_results[label] = analyse_prompt(
            model, analyzer, text, label,
            device=args.device,
            active_fraction=args.active_fraction,
        )

    # ── Output ─────────────────────────────────────────────────────────────────
    frac_tag     = f"top{int(args.active_fraction*100)}"
    heatmap_path = os.path.join(args.output_dir,
                                f"token_heatmap_{frac_tag}.png")
    json_path    = os.path.join(args.output_dir,
                                f"token_heatmap_{frac_tag}.json")

    print_summary(all_results, ana_metrics, args.active_fraction)
    plot_heatmap(all_results, ana_metrics, heatmap_path)
    plot_el_timeseries(all_results, ana_metrics, heatmap_path)
    save_json(all_results, ana_metrics, args.model, args.active_fraction, json_path)

    log.info("")
    log.info("Outputs:")
    log.info("  Heatmap    : %s", heatmap_path)
    log.info("  Timeseries : %s", heatmap_path.replace(".png", "_timeseries.png"))
    log.info("  JSON       : %s", json_path)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
