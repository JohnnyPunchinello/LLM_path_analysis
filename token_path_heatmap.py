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
  3. Thresholds at the 80th percentile → top-20% active subgraph G_t
  4. Runs Algorithm 1 (path DP) on G_t to get empirical E[L]_t

Hypothesis tested
-----------------
  Long residual paths are recruited specifically for COMPLEX reasoning tokens
  (e.g. the "=" in a chain-of-thought step). Simple grammatical tokens
  should cluster near short-path operation.

  Metric: "Recruitment Delta" = max_E[L](reasoning) − max_E[L](grammar)
  A positive delta confirms the hypothesis.

Outputs
-------
  results/token_heatmap.png       — side-by-side token heatmaps
  results/token_heatmap.json      — full token/E[L] data for further analysis

Usage
-----
  # Default: Pythia-2.8b, GPU
  python token_path_heatmap.py

  # Smaller model (quicker, CPU-feasible)
  python token_path_heatmap.py --model EleutherAI/pythia-1b --device cpu

  # Custom prompts
  python token_path_heatmap.py \\
      --simple  "The cat sat on the mat and then it" \\
      --complex "Q: What is 7 times 8 minus 6? A: 7*8=56, 56-6="
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
PROMPTS = {
    "grammar": (
        "The quick brown fox jumps over the lazy dog and then it"
    ),
    "reasoning": (
        "Question: If I have 3 sets of 4 apples and I eat 2, "
        "how many are left? "
        "Answer: I have 3 * 4 = 12 apples. 12 - 2 ="
    ),
}

# Top-N% active subgraph threshold
ACTIVE_FRACTION = 0.20          # keep top 20 % highest-attribution edges
EPSILON_QUANTILE = 1.0 - ACTIVE_FRACTION   # → 80th percentile threshold


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
    Line plot of E[L]_t vs token index for both prompts.
    Makes the recruitment trend and its magnitude easier to read numerically.
    """
    colors = {"grammar": "#2196F3", "reasoning": "#E53935"}
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

    for label, r in results.items():
        toks = r["token_strs"]
        el   = r["el_values"]
        xs   = np.arange(len(toks))
        # Mask BOS NaN
        mask = [not np.isnan(v) for v in el]
        ax.plot([x for x, m in zip(xs, mask) if m],
                [v for v, m in zip(el, mask) if m],
                color=colors.get(label, "gray"),
                lw=2, marker="o", ms=5,
                label=f"{label.title()}  (max={r['max_el']:.2f})")

        # Annotate the peak token
        if any(mask):
            peak_i = int(np.nanargmax(el))
            ax.annotate(
                f"  \"{_clean_tok(toks[peak_i])}\"",
                xy=(peak_i, el[peak_i]),
                fontsize=8.5, color=colors.get(label, "gray"), va="bottom")

    ax.axhline(ana_metrics.mean_path_length, color="#1a1aff",
               ls="--", lw=1.5, alpha=0.7,
               label=f"Analytical E[L] = {ana_metrics.mean_path_length:.1f}")
    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Empirical E[L]_t  (bits)", fontsize=11)
    ax.set_title("Path Recruitment by Token Position", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(bottom=0)

    ts_path = output_path.replace(".png", "_timeseries.png")
    fig.savefig(ts_path, dpi=150, bbox_inches="tight")
    log.info("Time-series saved → %s", ts_path)
    plt.close(fig)


# =============================================================================
# Output summary + JSON
# =============================================================================

def print_summary(results: Dict[str, Dict], ana_metrics, active_fraction: float) -> None:
    print()
    print("=" * 68)
    print("  TOKEN-WISE PATH RECRUITMENT SUMMARY")
    print("=" * 68)
    print(f"  Analytical E[L]       : {ana_metrics.mean_path_length:.4f}")
    print(f"  Active subgraph       : top {int(active_fraction*100)}% edges by AtP magnitude")
    print()

    for label, r in results.items():
        print(f"  [{label.upper()} prompt]")
        print(f"    Tokens              : {len(r['token_strs'])}")
        print(f"    E[L] range          : {r['min_el']:.2f} – {r['max_el']:.2f}")
        print(f"    Mean E[L]           : {r['mean_el']:.2f}")
        print(f"    Top-5 recruited tokens:")
        for rank, (pos, el) in enumerate(r["top5_positions"], 1):
            tok = _clean_tok(r["token_strs"][pos])
            predicted = _clean_tok(r["pred_tokens"][pos])
            print(f"      {rank}.  pos={pos:3d}  E[L]={el:.3f}  "
                  f"token='{tok}'  predicts='{predicted}'")
        print()

    # Recruitment delta
    if "grammar" in results and "reasoning" in results:
        delta = results["reasoning"]["max_el"] - results["grammar"]["max_el"]
        print(f"  Recruitment Delta  (reasoning max − grammar max)")
        print(f"    ΔE[L] = {results['reasoning']['max_el']:.3f} − "
              f"{results['grammar']['max_el']:.3f} = {delta:+.3f}")
        if delta > 0:
            print(f"    ✓  Reasoning tokens recruit LONGER paths  (+{delta:.3f} units)")
        else:
            print(f"    ✗  No clear recruitment difference at this threshold")
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

    if "grammar" in results and "reasoning" in results:
        payload["recruitment_delta"] = round(
            results["reasoning"]["max_el"] - results["grammar"]["max_el"], 4)

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
    parser.add_argument("--simple",  default=PROMPTS["grammar"],
                        help="Simple / grammar prompt")
    parser.add_argument("--complex", default=PROMPTS["reasoning"],
                        help="Complex / reasoning prompt")
    parser.add_argument("--active_fraction", type=float, default=ACTIVE_FRACTION,
                        help="Fraction of top-attributed edges to keep (default: 0.20)")
    parser.add_argument("--output_dir", default="results",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    model = load_model(args.model, device=args.device)

    # ── PathAnalyzer ───────────────────────────────────────────────────────────
    from path_analyzer import PathAnalyzer
    analyzer    = PathAnalyzer(model)
    arch_info   = analyzer.architecture_summary()
    ana_metrics = analyzer.analytical_path_distribution()

    log.info("Architecture : %s", arch_info["architecture"])
    log.info("n_layers     : %d", arch_info["n_layers"])
    log.info("max_path_len : %d", ana_metrics.max_path_len)
    log.info("Ana. H       : %.4f bits", ana_metrics.entropy)
    log.info("Ana. E[L]    : %.4f",      ana_metrics.mean_path_length)

    # ── Analyse both prompts ────────────────────────────────────────────────────
    prompts = {
        "grammar":   args.simple,
        "reasoning": args.complex,
    }
    all_results: Dict[str, Dict] = {}
    for label, text in prompts.items():
        all_results[label] = analyse_prompt(
            model, analyzer, text, label,
            device=args.device,
            active_fraction=args.active_fraction,
        )

    # ── Output ─────────────────────────────────────────────────────────────────
    heatmap_path = os.path.join(args.output_dir, "token_heatmap.png")
    json_path    = os.path.join(args.output_dir, "token_heatmap.json")

    print_summary(all_results, ana_metrics, args.active_fraction)
    plot_heatmap(all_results, ana_metrics, heatmap_path)
    plot_el_timeseries(all_results, ana_metrics, heatmap_path)
    save_json(all_results, ana_metrics, args.model, args.active_fraction, json_path)

    log.info("")
    log.info("Outputs:")
    log.info("  Heatmap    : %s", heatmap_path)
    log.info("  Timeseries : %s",
             heatmap_path.replace(".png", "_timeseries.png"))
    log.info("  JSON       : %s", json_path)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
