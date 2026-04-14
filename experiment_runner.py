#!/usr/bin/env python3
"""
experiment_runner.py — Path Distribution Theory: Experimental Loop
====================================================================

For each (model, task) combination:
  1. Compute Analytical Path Entropy  H_ana   (full architecture)
  2. Compute Empirical  Path Entropy  H_emp   (AtP-filtered active subgraph)
  3. Compute Synergy Gap              H_ana − H_emp
  4. Compute Tail-Mass Ratio          τ_k
  5. Compute task Accuracy

Outputs
  results/path_metrics.csv            — full results table
  results/entropy_vs_complexity.png   — 3-panel figure

Usage
-----
  python experiment_runner.py [options]

  # Quick test with GPT-2 on CPU:
  python experiment_runner.py --models gpt2 --n_samples 5 --device cpu

  # Full A100 run:
  python experiment_runner.py \\
      --models "meta-llama/Llama-3-8B,EleutherAI/gpt-j-6b,tiiuae/falcon-7b" \\
      --tasks  sst2,boolq,gsm8k  --n_samples 50
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # headless-safe; swapped to TkAgg below if interactive
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS: List[str] = [
    "meta-llama/Llama-3-8B",
    "EleutherAI/gpt-j-6b",
    "tiiuae/falcon-7b",
]

TASKS: Dict[str, Dict[str, Any]] = {
    "sst2":  {"hf_name": "sst2",  "hf_split": "validation", "complexity": 0},
    "boolq": {"hf_name": "boolq", "hf_split": "validation", "complexity": 1},
    "gsm8k": {"hf_name": "gsm8k", "hf_split": "test",       "complexity": 2,
              "hf_config": "main"},
}

COMPLEXITY_LABELS = {0: "SST-2 (Low)", 1: "BoolQ (Medium)", 2: "GSM8K (High)"}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Load a HookedTransformer.

    Tries 8-bit quantisation (bitsandbytes) first; falls back to fp16.
    """
    from transformer_lens import HookedTransformer

    log.info("Loading model: %s", model_name)

    # ── Attempt 8-bit via HuggingFace + bitsandbytes ──
    try:
        import bitsandbytes          # noqa: F401
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        log.info("  → attempting 8-bit quantised load …")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            dtype=torch.float16,
            move_to_device=False,
        )
        log.info("  ✓ 8-bit load succeeded.")
        model.eval()
        return model

    except Exception as exc:
        log.warning("  8-bit load failed (%s) — falling back to fp16.", exc)

    # ── fp16 fallback ──
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=dtype,
        device=device,
    )
    log.info("  ✓ fp16 load succeeded.")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_task_samples(
    task_name: str,
    n_samples: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load n_samples from a HuggingFace dataset and format as prompt-label pairs.
    """
    from datasets import load_dataset as hf_load

    cfg = TASKS[task_name]
    kwargs: Dict[str, Any] = {"path": cfg["hf_name"], "split": cfg["hf_split"]}
    if "hf_config" in cfg:
        kwargs["name"] = cfg["hf_config"]

    ds = hf_load(**kwargs)
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    samples = []
    for idx in indices.tolist():
        row    = ds[int(idx)]
        sample = _format_sample(task_name, row)
        if sample is not None:
            samples.append(sample)

    log.info("  Loaded %d samples for %s", len(samples), task_name)
    return samples


def _format_sample(task_name: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a dataset row into {prompt, label, task, correct_class}."""
    if task_name == "sst2":
        prompt = (
            f"Review: {row['sentence']}\n"
            f"Sentiment (positive/negative):"
        )
        label  = "positive" if row["label"] == 1 else "negative"
        return {"prompt": prompt, "label": label, "task": task_name,
                "correct_class": int(row["label"])}

    if task_name == "boolq":
        prompt = (
            f"Passage: {row['passage'][:512]}\n"
            f"Question: {row['question']}\n"
            f"Answer (yes/no):"
        )
        label  = "yes" if row["answer"] else "no"
        return {"prompt": prompt, "label": label, "task": task_name,
                "correct_class": int(row["answer"])}

    if task_name == "gsm8k":
        prompt = (
            f"Problem: {row['question']}\n"
            f"Solution: Let me solve this step by step.\n"
        )
        answer_str = row.get("answer", "")
        match      = re.search(r"####\s*([\d,.\-]+)", answer_str)
        label      = match.group(1).replace(",", "") if match else ""
        return {"prompt": prompt, "label": label, "task": task_name,
                "correct_class": None}

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_top_token(model, logits: torch.Tensor, pos: int = -1) -> str:
    """Return the decoded string of the highest-probability next token."""
    top_id = int(logits[0, pos].argmax())
    return model.tokenizer.decode([top_id]).strip().lower()


def check_accuracy(
    task_name:     str,
    model,
    tokens:        torch.Tensor,
    sample:        Dict[str, Any],
) -> bool:
    """Return True if the model's greedy next-token prediction is correct."""
    with torch.no_grad():
        logits = model(tokens, return_type="logits")

    pred = _decode_top_token(model, logits)

    if task_name == "sst2":
        return sample["label"].lower() in pred or pred in sample["label"].lower()

    if task_name == "boolq":
        return sample["label"].lower() in pred or pred in sample["label"].lower()

    if task_name == "gsm8k":
        expected = sample["label"]
        if not expected:
            return False
        try:
            nums = re.findall(r"\d+(?:\.\d+)?", pred)
            if nums:
                return abs(float(nums[0]) - float(expected)) < 1e-4
        except ValueError:
            pass
        return False

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model_names:       List[str],
    task_names:        List[str],
    n_samples:         int   = 50,
    output_dir:        str   = "results",
    device:            str   = "cuda",
    epsilon_quantile:  float = 0.25,
    seed:              int   = 42,
) -> List[Dict[str, Any]]:
    """
    Core experiment loop.

    For each (model, task):
      — compute attribution scores on each sample
      — set a task-level ε from the empirical distribution of scores
      — re-run path-count DP for every sample with that ε
      — aggregate metrics and accuracy

    Returns a list of result dicts (one per model×task combination).
    """
    from path_analyzer import PathAnalyzer, _to_metrics

    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for model_name in model_names:
        # ── load model ──
        try:
            model = load_model(model_name, device=device)
        except Exception as exc:
            log.error("Could not load %s: %s", model_name, exc)
            continue

        analyzer  = PathAnalyzer(model)
        arch_info = analyzer.architecture_summary()
        log.info("Architecture: %s", arch_info)

        # analytical entropy is constant for the architecture
        ana_metrics = analyzer.analytical_path_distribution()
        log.info(
            "  Analytical  H=%.4f bits  E[L]=%.2f  τ=%.4f",
            ana_metrics.entropy, ana_metrics.mean_path_length, ana_metrics.tail_mass_ratio,
        )

        for task_name in task_names:
            log.info("")
            log.info("─── Model=%-35s  Task=%s ───", model_name, task_name)

            # ── load samples ──
            try:
                samples = load_task_samples(task_name, n_samples=n_samples, seed=seed)
            except Exception as exc:
                log.error("  Failed to load task %s: %s", task_name, exc)
                continue

            # ── pass 1: attribution scores per sample ──
            per_attn: List[np.ndarray] = []
            per_mlp:  List[np.ndarray] = []
            per_corr: List[bool]       = []

            for i, sample in enumerate(samples):
                try:
                    tokens = model.to_tokens(
                        sample["prompt"], prepend_bos=True
                    ).to(device)

                    # accuracy
                    corr = check_accuracy(task_name, model, tokens, sample)
                    per_corr.append(corr)

                    # attribution
                    attn_sc, mlp_sc = analyzer.compute_attribution_scores(tokens)
                    per_attn.append(attn_sc.cpu().float().numpy())
                    per_mlp.append(mlp_sc.cpu().float().numpy())

                    if (i + 1) % 10 == 0:
                        log.info("  … %d/%d samples processed", i + 1, len(samples))

                except Exception as exc:
                    log.warning("  Sample %d failed (pass 1): %s", i, exc)

            if not per_attn:
                log.warning("  No valid samples for %s — skipping.", task_name)
                continue

            # ── task-level epsilon ──
            all_scores = np.concatenate(
                [np.concatenate(per_attn), np.concatenate(per_mlp)]
            )
            epsilon = float(np.quantile(all_scores, epsilon_quantile))
            log.info("  Task-level ε = %.2e  (q=%.2f)", epsilon, epsilon_quantile)

            # ── pass 2: empirical path metrics per sample ──
            emp_H:   List[float] = []
            emp_mu:  List[float] = []
            emp_tau: List[float] = []

            for i in range(len(per_attn)):
                try:
                    active_attn = (per_attn[i] > epsilon).tolist()
                    active_mlp  = (per_mlp[i]  > epsilon).tolist()
                    counts      = analyzer._path_count_dp(active_attn, active_mlp)
                    metrics     = _to_metrics(counts)

                    emp_H.append(metrics.entropy)
                    emp_mu.append(metrics.mean_path_length)
                    emp_tau.append(metrics.tail_mass_ratio)

                except Exception as exc:
                    log.warning("  Sample %d failed (pass 2): %s", i, exc)

            if not emp_H:
                log.warning("  No empirical metrics for %s — skipping.", task_name)
                continue

            # ── aggregate ──
            accuracy  = float(np.mean(per_corr))  if per_corr  else 0.0
            mean_H    = float(np.mean(emp_H))
            std_H     = float(np.std(emp_H))  if len(emp_H) > 1 else 0.0
            mean_mu   = float(np.mean(emp_mu))
            finite_tau = [t for t in emp_tau if np.isfinite(t)]
            mean_tau  = float(np.mean(finite_tau)) if finite_tau else float("inf")

            result: Dict[str, Any] = {
                "model":                   model_name,
                "task":                    task_name,
                "task_complexity":         TASKS[task_name]["complexity"],
                "accuracy":                round(accuracy,           4),
                # analytical (architecture)
                "analytical_entropy":      round(ana_metrics.entropy,          4),
                "analytical_mean_path":    round(ana_metrics.mean_path_length, 4),
                "analytical_tail_ratio":   round(ana_metrics.tail_mass_ratio,  4),
                # empirical (active subgraph)
                "empirical_entropy":       round(mean_H,    4),
                "empirical_entropy_std":   round(std_H,     4),
                "empirical_mean_path":     round(mean_mu,   4),
                "empirical_tail_ratio":    round(mean_tau,  4),
                # gap
                "synergy_gap":             round(ana_metrics.entropy - mean_H, 4),
                # meta
                "n_layers":                arch_info["n_layers"],
                "architecture":            arch_info["architecture"],
                "epsilon":                 round(float(epsilon), 8),
                "epsilon_quantile":        epsilon_quantile,
                "n_samples_used":          len(emp_H),
            }
            results.append(result)

            log.info(
                "  Acc=%.3f  H_ana=%.3f  H_emp=%.3f±%.3f  Gap=%.3f  E[L]=%.2f",
                accuracy, ana_metrics.entropy, mean_H, std_H,
                result["synergy_gap"], mean_mu,
            )

        # free VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results: List[Dict[str, Any]], path: str) -> None:
    if not results:
        log.warning("No results — CSV not written.")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info("Results saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    results:     List[Dict[str, Any]],
    output_path: str = "results/entropy_vs_complexity.png",
) -> None:
    """
    3-panel figure:
      Left   — Empirical Path Entropy H(π̂) vs task complexity
      Centre — Synergy Gap (H_ana − H_emp) vs task complexity
      Right  — Mean Path Length E[L]  vs task complexity

    X-axis: 0=SST-2 (Low), 1=BoolQ (Medium), 2=GSM8K (High)
    One series per model, with ±1 std error bars on the left panel.
    """
    if not results:
        log.warning("No results — plot not generated.")
        return

    models       = sorted({r["model"]          for r in results})
    complexities = sorted({r["task_complexity"] for r in results})

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        "Path Distribution Theory  ·  Empirical Analysis",
        fontsize=15, fontweight="bold", y=1.02,
    )

    cmap    = plt.get_cmap("tab10")
    colors  = [cmap(i) for i in range(len(models))]
    markers = ["o", "s", "^", "D", "v", "P"]
    xticks  = complexities
    xlabels = [COMPLEXITY_LABELS[c] for c in complexities]

    def _series(model_rows, key):
        xs, ys = [], []
        for c in complexities:
            vals = [r[key] for r in model_rows if r["task_complexity"] == c]
            if vals:
                xs.append(c)
                ys.append(float(np.mean(vals)))
        return xs, ys

    # ── Panel 1: Empirical entropy ──
    ax = axes[0]
    for idx, mname in enumerate(models):
        rows = [r for r in results if r["model"] == mname]
        xs, ys = _series(rows, "empirical_entropy")
        yerr   = [np.mean([r["empirical_entropy_std"]
                           for r in rows if r["task_complexity"] == c])
                  for c in complexities if any(r["task_complexity"] == c for r in rows)]
        if xs:
            label = mname.split("/")[-1]
            ax.errorbar(xs, ys, yerr=yerr,
                        marker=markers[idx % len(markers)],
                        color=colors[idx], label=label,
                        capsize=4, linewidth=2, markersize=8)

    ax.set_xlabel("Task Complexity", fontsize=11)
    ax.set_ylabel("H(π̂)  [bits]", fontsize=11)
    ax.set_title("Empirical Path Entropy", fontsize=12)
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=12, ha="right")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Panel 2: Synergy gap ──
    ax = axes[1]
    for idx, mname in enumerate(models):
        rows = [r for r in results if r["model"] == mname]
        xs, ys = _series(rows, "synergy_gap")
        if xs:
            ax.plot(xs, ys,
                    marker=markers[idx % len(markers)],
                    color=colors[idx], label=mname.split("/")[-1],
                    linewidth=2, markersize=8)

    ax.set_xlabel("Task Complexity", fontsize=11)
    ax.set_ylabel("H_ana − H_emp  [bits]", fontsize=11)
    ax.set_title("Synergy Gap", fontsize=12)
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=12, ha="right")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)

    # ── Panel 3: Mean path length ──
    ax = axes[2]
    for idx, mname in enumerate(models):
        rows = [r for r in results if r["model"] == mname]
        xs, ys = _series(rows, "empirical_mean_path")
        if xs:
            ax.plot(xs, ys,
                    marker=markers[idx % len(markers)],
                    color=colors[idx], label=mname.split("/")[-1],
                    linewidth=2, markersize=8)

    # Add reference line: analytical mean path length (per model)
    for idx, mname in enumerate(models):
        rows = [r for r in results if r["model"] == mname]
        if rows:
            ana_mu = rows[0]["analytical_mean_path"]
            ax.axhline(
                ana_mu, color=colors[idx], linestyle=":",
                linewidth=1.2, alpha=0.6,
                label=f"{mname.split('/')[-1]} (ana.)",
            )

    ax.set_xlabel("Task Complexity", fontsize=11)
    ax.set_ylabel("E[L]", fontsize=11)
    ax.set_title("Mean Path Length  (dotted = analytical)", fontsize=12)
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=12, ha="right")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Plot saved → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    col_w = 28
    header = (
        f"{'Model':{col_w}} {'Task':8} {'Acc':6} "
        f"{'H_ana':7} {'H_emp':7} {'Gap':7} {'E[L]':6} {'τ':7}"
    )
    print()
    print("=" * len(header))
    print("  PATH DISTRIBUTION THEORY — RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print("─" * len(header))
    for r in results:
        print(
            f"{r['model'].split('/')[-1]:{col_w}} "
            f"{r['task']:8} "
            f"{r['accuracy']:6.3f} "
            f"{r['analytical_entropy']:7.3f} "
            f"{r['empirical_entropy']:7.3f} "
            f"{r['synergy_gap']:7.3f} "
            f"{r['empirical_mean_path']:6.2f} "
            f"{r['empirical_tail_ratio']:7.4f}"
        )
    print("=" * len(header))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Path Distribution Theory — mechanistic interpretability pipeline"
    )
    p.add_argument(
        "--models", type=str, default=",".join(DEFAULT_MODELS),
        help="Comma-separated HuggingFace model IDs (default: Llama-3-8B, GPT-J-6B, Falcon-7B)",
    )
    p.add_argument(
        "--tasks", type=str, default=",".join(TASKS.keys()),
        help="Comma-separated task names: sst2,boolq,gsm8k  (default: all three)",
    )
    p.add_argument(
        "--n_samples", type=int, default=50,
        help="Samples per task  (default: 50)",
    )
    p.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory for CSV and plot outputs  (default: results/)",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device  (default: cuda if available)",
    )
    p.add_argument(
        "--epsilon_quantile", type=float, default=0.25,
        help="Quantile of attribution scores used to set ε  (default: 0.25 → top-75%% active)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset shuffling  (default: 42)",
    )
    p.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for fp-fallback load  (default: float16)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    task_names  = [t.strip() for t in args.tasks.split(",")  if t.strip()]

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }

    log.info("Device    : %s", args.device)
    log.info("Models    : %s", model_names)
    log.info("Tasks     : %s", task_names)
    log.info("Samples   : %d per task", args.n_samples)
    log.info("ε-quantile: %.2f", args.epsilon_quantile)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        log.info("GPU       : %s  (%.1f GB VRAM)", prop.name, prop.total_memory / 1e9)

    results = run_experiment(
        model_names=model_names,
        task_names=task_names,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        device=args.device,
        epsilon_quantile=args.epsilon_quantile,
        seed=args.seed,
    )

    csv_path  = os.path.join(args.output_dir, "path_metrics.csv")
    plot_path = os.path.join(args.output_dir, "entropy_vs_complexity.png")

    save_csv(results, csv_path)
    plot_results(results, plot_path)
    print_summary(results)


if __name__ == "__main__":
    main()
