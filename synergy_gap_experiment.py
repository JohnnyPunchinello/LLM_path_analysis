#!/usr/bin/env python3
"""
synergy_gap_experiment.py
=========================
Empirical test of the Synergy Gap hypothesis across diverse tasks and
architectures with residual connections.

Hypothesis
----------
  The Synergy Gap  ΔH = H_ana − H_emp  (analytical path entropy minus
  empirical active-subgraph entropy) DECREASES as task complexity increases.
  Harder tasks force models to route activations through more of the
  available residual-path space, leaving less entropy "unused".

Experiment design
-----------------
  12 tasks  ×  3 complexity levels  ×  4 architecture families
  ┌────────────────────────────────────────────────────────────┐
  │  Low (0)    : SST-2, PIQA, CoLA                           │
  │  Medium (1) : BoolQ, ARC-Easy, WinoGrande, COPA,          │
  │               OpenBookQA                                   │
  │  High (2)   : HellaSwag, ARC-Challenge, LAMBADA, GSM8K    │
  └────────────────────────────────────────────────────────────┘

  Architecture families (all use residual connections):
    pythia  — Pre-RMSNorm, sequential  (6 sizes, 70M → 6.9B)
    gpt2    — Pre-LayerNorm, sequential (4 sizes, 117M → 1.5B)
    neo     — Pre-LayerNorm, sequential (GPT-Neo) +
              Parallel ternary (GPT-J)              ← architecture contrast
    opt     — Post-LayerNorm, sequential (4 sizes)  ← normalization contrast

Outputs
-------
  results/synergy_gap_results.csv
  results/fig1_gap_vs_complexity.png   — main paper figure
  results/fig2_arch_comparison.png     — architecture-type boxplot
  results/fig3_scaling.png             — scaling curves (log params vs ΔH)
  results/synergy_gap_report.md        — text summary with stats

Usage
-----
  # Minimal CPU test (GPT-2 small + small model, 3 tasks, 5 samples each)
  python synergy_gap_experiment.py --model_group gpt2 \\
      --tasks sst2,boolq,hellaswag --n_samples 5 --device cpu

  # Pythia scaling study (GPU recommended)
  python synergy_gap_experiment.py --model_group pythia \\
      --tasks sst2,piqa,boolq,arc_easy,copa,hellaswag,lambada --n_samples 50

  # Full architecture comparison
  python synergy_gap_experiment.py --model_group neo \\
      --tasks sst2,boolq,arc_easy,hellaswag,arc_challenge --n_samples 50

  # OPT family (post-LN normalization comparison)
  python synergy_gap_experiment.py --model_group opt \\
      --tasks sst2,cola,boolq,arc_easy,hellaswag --n_samples 50

  # Load existing CSV and just regenerate figures
  python synergy_gap_experiment.py --plot_only results/synergy_gap_results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.4})

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not found — Pearson p-values will be omitted")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Model catalogue
# =============================================================================

MODEL_GROUPS: Dict[str, List[str]] = {
    # Pre-RMSNorm sequential — scaling study
    "pythia": [
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b",
    ],
    # Pre-LayerNorm sequential — scaling study
    "gpt2": [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
    ],
    # Pre-LN sequential (GPT-Neo) + parallel ternary (GPT-J)
    # Lets us isolate the sequential/parallel architecture difference
    "neo": [
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/gpt-j-6b",         # parallel_attn_mlp=True
    ],
    # Post-LayerNorm sequential — normalization contrast with Pythia/GPT-2
    "opt": [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
    ],
    # Quick smoke-test group (CPU-feasible, small models only)
    "small": [
        "EleutherAI/pythia-70m",
        "gpt2",
        "EleutherAI/gpt-neo-125m",
        "facebook/opt-125m",
    ],
}
MODEL_GROUPS["all"] = (
    MODEL_GROUPS["pythia"]
    + MODEL_GROUPS["gpt2"]
    + MODEL_GROUPS["neo"]
    + MODEL_GROUPS["opt"]
)

MODEL_PARAMS_B: Dict[str, float] = {
    "EleutherAI/pythia-70m":    0.070,
    "EleutherAI/pythia-160m":   0.160,
    "EleutherAI/pythia-410m":   0.410,
    "EleutherAI/pythia-1b":     1.0,
    "EleutherAI/pythia-2.8b":   2.8,
    "EleutherAI/pythia-6.9b":   6.9,
    "gpt2":                     0.117,
    "gpt2-medium":              0.345,
    "gpt2-large":               0.774,
    "gpt2-xl":                  1.558,
    "EleutherAI/gpt-neo-125m":  0.125,
    "EleutherAI/gpt-neo-1.3B":  1.3,
    "EleutherAI/gpt-neo-2.7B":  2.7,
    "EleutherAI/gpt-j-6b":      6.0,
    "facebook/opt-125m":        0.125,
    "facebook/opt-350m":        0.350,
    "facebook/opt-1.3b":        1.3,
    "facebook/opt-2.7b":        2.7,
}

# Family name that appears in plots  (model_name → display family)
def _infer_family(model_name: str) -> str:
    m = model_name.lower()
    if "pythia"   in m:  return "Pythia (Pre-RMSNorm, sequential)"
    if "gpt-j"    in m:  return "GPT-J (parallel ternary)"
    if "gpt-neo"  in m:  return "GPT-Neo (Pre-LN, sequential)"
    if "gpt2"     in m:  return "GPT-2 (Pre-LN, sequential)"
    if "opt"      in m:  return "OPT (Post-LN, sequential)"
    if "llama"    in m:  return "LLaMA (Pre-RMS, sequential)"
    if "falcon"   in m:  return "Falcon (parallel, sequential)"
    return "Other"

def _arch_type(model_name: str) -> str:
    """Coarse architecture type for grouped comparison."""
    m = model_name.lower()
    if "gpt-j" in m or "falcon" in m:  return "parallel"
    if "opt"   in m:                    return "sequential-post-LN"
    return "sequential-pre-LN"

def _short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


# =============================================================================
# Task catalogue
# =============================================================================
# complexity: 0=Low  1=Medium  2=High
# eval_type:  binary | choice | last_word | generative

TASKS: Dict[str, Dict[str, Any]] = {
    # ── Low complexity (0) ──────────────────────────────────────────────────
    "sst2": {
        "display": "SST-2", "hf_name": "sst2", "hf_split": "validation",
        "complexity": 0, "eval_type": "binary",
        "labels": [["negative", "Negative", "bad"], ["positive", "Positive", "good"]],
        "description": "Sentiment classification (positive / negative)",
    },
    "piqa": {
        "display": "PIQA", "hf_name": "piqa", "hf_split": "validation",
        "complexity": 0, "eval_type": "choice",
        "description": "Physical intuition: pick the correct solution",
    },
    "cola": {
        "display": "CoLA", "hf_name": "glue", "hf_split": "validation",
        "hf_config": "cola",
        "complexity": 0, "eval_type": "binary",
        "labels": [["unacceptable", "incorrect", "wrong"],
                   ["acceptable",   "correct",   "grammatical"]],
        "description": "Linguistic acceptability (grammatical / ungrammatical)",
    },
    # ── Medium complexity (1) ────────────────────────────────────────────────
    "boolq": {
        "display": "BoolQ", "hf_name": "boolq", "hf_split": "validation",
        "complexity": 1, "eval_type": "binary",
        "labels": [["no", "No", "false", "False"], ["yes", "Yes", "true", "True"]],
        "description": "Boolean QA over short passages",
    },
    "arc_easy": {
        "display": "ARC-Easy", "hf_name": "ai2_arc", "hf_split": "validation",
        "hf_config": "ARC-Easy",
        "complexity": 1, "eval_type": "choice",
        "description": "Elementary science QA (easy subset)",
    },
    "winogrande": {
        "display": "WinoGrande", "hf_name": "winogrande", "hf_split": "validation",
        "hf_config": "winogrande_xl",
        "complexity": 1, "eval_type": "choice",
        "description": "Commonsense coreference resolution",
    },
    "copa": {
        "display": "COPA", "hf_name": "super_glue", "hf_split": "validation",
        "hf_config": "copa",
        "complexity": 1, "eval_type": "choice",
        "description": "Causal reasoning: pick correct cause / effect",
    },
    "openbookqa": {
        "display": "OpenBookQA", "hf_name": "openbookqa", "hf_split": "validation",
        "hf_config": "main",
        "complexity": 1, "eval_type": "choice",
        "description": "Elementary science QA requiring world knowledge",
    },
    # ── High complexity (2) ──────────────────────────────────────────────────
    "hellaswag": {
        "display": "HellaSwag", "hf_name": "hellaswag", "hf_split": "validation",
        "complexity": 2, "eval_type": "choice",
        "description": "Sentence completion requiring grounded commonsense",
    },
    "arc_challenge": {
        "display": "ARC-Challenge", "hf_name": "ai2_arc", "hf_split": "validation",
        "hf_config": "ARC-Challenge",
        "complexity": 2, "eval_type": "choice",
        "description": "Elementary science QA (harder subset, adversarially filtered)",
    },
    "lambada": {
        "display": "LAMBADA", "hf_name": "lambada_openai", "hf_split": "test",
        "complexity": 2, "eval_type": "last_word",
        "description": "Long-range language modelling: predict the final word",
    },
    "gsm8k": {
        "display": "GSM8K", "hf_name": "gsm8k", "hf_split": "test",
        "hf_config": "main",
        "complexity": 2, "eval_type": "generative",
        "description": "Grade-school math word problems (multi-step reasoning)",
    },
}

COMPLEXITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}

DEFAULT_TASKS   = ["sst2", "piqa", "boolq", "arc_easy", "copa",
                   "hellaswag", "arc_challenge", "lambada"]
DEFAULT_MODELS  = MODEL_GROUPS["small"]


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_name: str, device: str = "cuda",
               dtype: torch.dtype = torch.float16):
    from transformer_lens import HookedTransformer

    log.info("Loading  %s …", model_name)
    # Try 8-bit quantised first (large models on single GPU)
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

    model = HookedTransformer.from_pretrained(model_name, dtype=dtype, device=device)
    model.eval()
    log.info("  -> fp16 load OK")
    return model


# =============================================================================
# Dataset loading
# =============================================================================

def load_samples(task_name: str, n_samples: int = 50, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset as _load

    cfg = TASKS[task_name]
    kwargs: Dict[str, Any] = {"path": cfg["hf_name"], "split": cfg["hf_split"]}
    if "hf_config" in cfg:
        kwargs["name"] = cfg["hf_config"]

    ds      = _load(**kwargs)
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    samples = []
    for idx in indices.tolist():
        row = ds[int(idx)]
        s   = _format_sample(task_name, row)
        if s is not None:
            samples.append(s)

    log.info("  Loaded %d/%d samples for %s", len(samples), n_samples, task_name)
    return samples


def _format_sample(task_name: str, row: Dict) -> Optional[Dict]:
    base = {"task": task_name}

    # ── SST-2 ────────────────────────────────────────────────────────────────
    if task_name == "sst2":
        return {**base,
            "prompt": f"Review: {row['sentence']}\nSentiment:",
            "label": "positive" if row["label"] == 1 else "negative",
            "correct_class": int(row["label"])}

    # ── PIQA ─────────────────────────────────────────────────────────────────
    if task_name == "piqa":
        return {**base,
            "prompt": f"Goal: {row['goal']}\nHow to achieve this:",
            "choices": [f" {row['sol1']}", f" {row['sol2']}"],
            "correct_class": int(row["label"])}

    # ── CoLA ─────────────────────────────────────────────────────────────────
    if task_name == "cola":
        return {**base,
            "prompt": (f"Sentence: \"{row['sentence']}\"\n"
                       f"Is this sentence grammatically acceptable?"),
            "label": "acceptable" if row["label"] == 1 else "unacceptable",
            "correct_class": int(row["label"])}

    # ── BoolQ ────────────────────────────────────────────────────────────────
    if task_name == "boolq":
        return {**base,
            "prompt": (f"Passage: {row['passage'][:400]}\n"
                       f"Question: {row['question']}\nAnswer (yes/no):"),
            "label": "yes" if row["answer"] else "no",
            "correct_class": int(row["answer"])}

    # ── ARC (Easy + Challenge) ───────────────────────────────────────────────
    if task_name in ("arc_easy", "arc_challenge"):
        ctext = row["choices"]["text"]
        clabel = row["choices"]["label"]
        ans = row["answerKey"]
        correct_idx = clabel.index(ans) if ans in clabel else 0
        return {**base,
            "prompt": f"Question: {row['question']}\nAnswer:",
            "choices": [f" {c}" for c in ctext],
            "correct_class": correct_idx}

    # ── WinoGrande ───────────────────────────────────────────────────────────
    if task_name == "winogrande":
        blank = row["sentence"].find("_")
        ctx   = row["sentence"][:blank]
        return {**base,
            "prompt": ctx,
            "choices": [row["option1"], row["option2"]],
            "correct_class": int(row["answer"]) - 1}

    # ── COPA ─────────────────────────────────────────────────────────────────
    if task_name == "copa":
        conn = "because" if row["question"] == "cause" else "therefore"
        prompt = f"{row['premise'].rstrip('.')} {conn}"
        return {**base,
            "prompt": prompt,
            "choices": [f" {row['choice1'].rstrip('.')}", f" {row['choice2'].rstrip('.')}"],
            "correct_class": int(row["label"])}

    # ── OpenBookQA ───────────────────────────────────────────────────────────
    if task_name == "openbookqa":
        ctext  = row["choices"]["text"]
        clabel = row["choices"]["label"]
        ans    = row["answerKey"]
        correct_idx = clabel.index(ans) if ans in clabel else 0
        return {**base,
            "prompt": f"Question: {row['question_stem']}\nAnswer:",
            "choices": [f" {c}" for c in ctext],
            "correct_class": correct_idx}

    # ── HellaSwag ────────────────────────────────────────────────────────────
    if task_name == "hellaswag":
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        return {**base,
            "prompt": ctx,
            "choices": [" " + e for e in row["endings"]],
            "correct_class": int(row["label"])}

    # ── LAMBADA ──────────────────────────────────────────────────────────────
    if task_name == "lambada":
        text  = row["text"].strip()
        # Split off the last word; model must predict it
        parts = text.rsplit(" ", 1)
        if len(parts) < 2:
            return None
        prompt, last_word = parts
        return {**base,
            "prompt": prompt,
            "label": last_word,
            "correct_class": None}

    # ── GSM8K ────────────────────────────────────────────────────────────────
    if task_name == "gsm8k":
        m = re.search(r"####\s*([\d,.\-]+)", row.get("answer", ""))
        label = m.group(1).replace(",", "") if m else ""
        return {**base,
            "prompt": f"Problem: {row['question']}\nSolution:",
            "label": label,
            "correct_class": None}

    return None


# =============================================================================
# Accuracy evaluation (log-probability based)
# =============================================================================

def _token_ids_for_words(model, words: List[str]) -> List[int]:
    """All single-token IDs for word variants (bare, capitalised, space-prefixed)."""
    variants: List[str] = []
    for w in words:
        variants += [w, w.capitalize(), w.upper(), f" {w}", f" {w.capitalize()}"]
    ids: List[int] = []
    for v in variants:
        enc = model.tokenizer.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return list(set(ids))


def _mean_logprob_continuation(model, prompt_tok: torch.Tensor,
                                cont_tok: torch.Tensor) -> float:
    """Mean per-token log-probability of cont_tok given prompt_tok."""
    if cont_tok.shape[-1] == 0:
        return -float("inf")
    full = torch.cat([prompt_tok, cont_tok], dim=-1)
    with torch.no_grad():
        logits = model(full, return_type="logits")
    lp      = torch.log_softmax(logits[0], dim=-1)
    n_p     = prompt_tok.shape[-1]
    n_c     = cont_tok.shape[-1]
    score   = sum(lp[n_p + i - 1, cont_tok[0, i]].item() for i in range(n_c))
    return score / n_c


def check_accuracy(task_name: str, model, tokens: torch.Tensor,
                   sample: Dict, device: str) -> bool:
    eval_type = TASKS[task_name]["eval_type"]

    # ── Binary label comparison ──────────────────────────────────────────────
    if eval_type == "binary":
        label_groups = TASKS[task_name]["labels"]
        with torch.no_grad():
            logits = model(tokens, return_type="logits")
        last   = logits[0, -1]
        scores = []
        for words in label_groups:
            ids = _token_ids_for_words(model, words)
            scores.append(float(last[ids].max()) if ids else -float("inf"))
        return int(np.argmax(scores)) == sample["correct_class"]

    # ── Multiple-choice continuation scoring ─────────────────────────────────
    if eval_type == "choice":
        choices = sample.get("choices", [])
        if not choices:
            return False
        scores = []
        for ch in choices:
            enc = model.tokenizer.encode(ch, add_special_tokens=False)
            c_t = torch.tensor([enc], dtype=torch.long, device=device)
            scores.append(_mean_logprob_continuation(model, tokens, c_t))
        return int(np.argmax(scores)) == sample["correct_class"]

    # ── LAMBADA: top-1 prediction == last word ───────────────────────────────
    if eval_type == "last_word":
        target = sample.get("label", "").strip().lower()
        if not target:
            return False
        with torch.no_grad():
            logits = model(tokens, return_type="logits")
        top_id  = int(logits[0, -1].argmax())
        decoded = model.tokenizer.decode([top_id]).strip().lower()
        return decoded == target

    # ── Generative (GSM8K) ───────────────────────────────────────────────────
    if eval_type == "generative":
        expected = sample.get("label", "")
        if not expected:
            return False
        with torch.no_grad():
            out = model.generate(tokens, max_new_tokens=25, do_sample=False)
        new_toks = out[0, tokens.shape[-1]:]
        decoded  = model.tokenizer.decode(new_toks, skip_special_tokens=True)
        nums = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", decoded)
        if nums:
            try:
                return abs(float(nums[0].replace(",", "")) - float(expected)) < 0.5
            except ValueError:
                pass
        return False

    return False


# =============================================================================
# Main experiment loop
# =============================================================================

def run_experiment(
    model_names:      List[str],
    task_names:       List[str],
    n_samples:        int   = 50,
    output_dir:       str   = "results",
    device:           str   = "cuda",
    epsilon_quantile: float = 0.25,
    seed:             int   = 42,
    resume_csv:       Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Two-pass experiment loop.

    Pass 1 — collect AtP scores + accuracy for every sample.
    Pass 2 — threshold at task-level ε, compute empirical path metrics.
    """
    from path_analyzer import PathAnalyzer, _to_metrics

    os.makedirs(output_dir, exist_ok=True)

    # Load previously computed results to allow resuming
    completed: set = set()
    results: List[Dict[str, Any]] = []
    if resume_csv and Path(resume_csv).exists():
        with open(resume_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(dict(row))
                completed.add((row["model"], row["task"]))
        log.info("Resumed from %s  (%d rows already done)", resume_csv, len(results))

    for model_name in model_names:
        # Skip entire model if all its tasks are done
        remaining_tasks = [t for t in task_names
                           if (model_name, t) not in completed]
        if not remaining_tasks:
            log.info("All tasks done for %s — skipping load.", model_name)
            continue

        try:
            model = load_model(model_name, device=device)
        except Exception as exc:
            log.error("Could not load %s: %s", model_name, exc)
            continue

        analyzer  = PathAnalyzer(model)
        arch_info = analyzer.architecture_summary()
        log.info("Architecture: %s", arch_info)

        ana_metrics = analyzer.analytical_path_distribution()
        log.info("  Analytical  H=%.4f bits  E[L]=%.2f  τ=%.4f  max_l=%d",
                 ana_metrics.entropy, ana_metrics.mean_path_length,
                 ana_metrics.tail_mass_ratio, ana_metrics.max_path_len)

        for task_name in remaining_tasks:
            if task_name not in TASKS:
                log.warning("Unknown task '%s' — skipping.", task_name)
                continue

            log.info("")
            log.info("─── Model=%-40s  Task=%s ───", model_name, task_name)

            try:
                samples = load_samples(task_name, n_samples=n_samples, seed=seed)
            except Exception as exc:
                log.error("  Could not load task %s: %s", task_name, exc)
                continue

            # ── Pass 1: AtP scores + accuracy ──────────────────────────────
            per_attn: List[np.ndarray] = []
            per_mlp:  List[np.ndarray] = []
            per_corr: List[bool]       = []

            for i, sample in enumerate(samples):
                try:
                    tokens = model.to_tokens(
                        sample["prompt"], prepend_bos=True).to(device)
                    if tokens.shape[-1] > 512:
                        tokens = tokens[:, -512:]

                    per_corr.append(
                        check_accuracy(task_name, model, tokens, sample, device))

                    a_sc, m_sc = analyzer.compute_attribution_scores(tokens)
                    per_attn.append(a_sc.cpu().float().numpy())
                    per_mlp.append(m_sc.cpu().float().numpy())

                    if (i + 1) % 10 == 0:
                        log.info("  … %d/%d", i + 1, len(samples))

                except Exception as exc:
                    log.warning("  Sample %d failed (pass 1): %s", i, exc)

            if not per_attn:
                log.warning("  No valid samples — skipping %s.", task_name)
                continue

            # ── Task-level ε ───────────────────────────────────────────────
            all_scores = np.concatenate(
                [np.concatenate(per_attn), np.concatenate(per_mlp)])
            epsilon = float(np.quantile(all_scores, epsilon_quantile))
            log.info("  ε = %.2e  (q=%.2f quantile)", epsilon, epsilon_quantile)

            # ── Pass 2: empirical path metrics ─────────────────────────────
            emp_H:   List[float] = []
            emp_mu:  List[float] = []
            emp_tau: List[float] = []
            emp_n_active: List[float] = []

            for i in range(len(per_attn)):
                try:
                    act_a = (per_attn[i] > epsilon).tolist()
                    act_m = (per_mlp[i]  > epsilon).tolist()
                    m     = _to_metrics(analyzer._path_count_dp(act_a, act_m))
                    emp_H.append(m.entropy)
                    emp_mu.append(m.mean_path_length)
                    emp_tau.append(m.tail_mass_ratio)
                    emp_n_active.append(
                        sum(act_a) + sum(act_m))
                except Exception as exc:
                    log.warning("  Sample %d failed (pass 2): %s", i, exc)

            if not emp_H:
                continue

            accuracy = float(np.mean(per_corr))
            mean_H   = float(np.mean(emp_H))
            std_H    = float(np.std(emp_H))  if len(emp_H) > 1 else 0.0
            mean_mu  = float(np.mean(emp_mu))
            finite_t = [t for t in emp_tau if np.isfinite(t)]
            mean_tau = float(np.mean(finite_t)) if finite_t else float("inf")
            mean_n_act = float(np.mean(emp_n_active))

            row: Dict[str, Any] = {
                "model":                model_name,
                "model_short":          _short_name(model_name),
                "model_params_B":       MODEL_PARAMS_B.get(model_name, -1),
                "family":               _infer_family(model_name),
                "arch_type":            _arch_type(model_name),
                "task":                 task_name,
                "task_display":         TASKS[task_name]["display"],
                "task_complexity":      TASKS[task_name]["complexity"],
                "task_complexity_label":COMPLEXITY_LABELS[TASKS[task_name]["complexity"]],
                "accuracy":             round(accuracy,  4),
                # analytical
                "analytical_entropy":   round(ana_metrics.entropy,          4),
                "analytical_mean_path": round(ana_metrics.mean_path_length, 4),
                "analytical_tail_ratio":round(ana_metrics.tail_mass_ratio,  4),
                # empirical
                "empirical_entropy":    round(mean_H,   4),
                "empirical_entropy_std":round(std_H,    4),
                "empirical_mean_path":  round(mean_mu,  4),
                "empirical_tail_ratio": round(mean_tau, 4),
                "mean_active_edges":    round(mean_n_act, 2),
                # derived
                "synergy_gap":          round(ana_metrics.entropy - mean_H, 4),
                # meta
                "n_layers":             arch_info["n_layers"],
                "architecture":         arch_info["architecture"],
                "normalization":        arch_info["normalization"],
                "max_path_len":         ana_metrics.max_path_len,
                "epsilon":              round(float(epsilon), 8),
                "epsilon_quantile":     epsilon_quantile,
                "n_samples_used":       len(emp_H),
            }
            results.append(row)

            log.info(
                "  Acc=%.3f  H_ana=%.3f  H_emp=%.3f±%.3f  Gap=%.4f  "
                "E[L]=%.2f  active_edges=%.1f",
                accuracy, ana_metrics.entropy, mean_H, std_H,
                row["synergy_gap"], mean_mu, mean_n_act)

            # Save after every (model, task) so we can resume
            _save_csv(results,
                      os.path.join(output_dir, "synergy_gap_results.csv"))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# =============================================================================
# CSV helpers
# =============================================================================

def _save_csv(results: List[Dict], path: str) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def load_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            # cast numeric columns
            for col in ("model_params_B", "task_complexity", "accuracy",
                        "analytical_entropy", "empirical_entropy",
                        "empirical_entropy_std", "synergy_gap",
                        "empirical_mean_path", "empirical_tail_ratio",
                        "mean_active_edges", "n_layers", "n_samples_used"):
                try:
                    row[col] = float(row[col])
                except (KeyError, ValueError):
                    pass
            rows.append(row)
    return rows


# =============================================================================
# Statistical analysis
# =============================================================================

def analyse(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute per-model Pearson r(complexity, synergy_gap) and trend labels.
    Also compute family-level and arch-type-level summaries.
    """
    from collections import defaultdict

    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    model_stats = []
    for model, rows in by_model.items():
        rows_s  = sorted(rows, key=lambda r: float(r["task_complexity"]))
        xs      = np.array([float(r["task_complexity"]) for r in rows_s])
        ys      = np.array([float(r["synergy_gap"])     for r in rows_s])
        if len(xs) < 2:
            continue
        if _HAS_SCIPY:
            r_val, p_val = _scipy_stats.pearsonr(xs, ys)
        else:
            r_val = float(np.corrcoef(xs, ys)[0, 1])
            p_val = None

        if r_val < -0.3 and (p_val is None or p_val < 0.1):
            trend = "Closing"
        elif r_val > 0.3 and (p_val is None or p_val < 0.1):
            trend = "Widening"
        else:
            trend = "Fixed"

        model_stats.append({
            "model":      model,
            "model_short":_short_name(model),
            "family":     rows[0].get("family", ""),
            "arch_type":  rows[0].get("arch_type", ""),
            "params_B":   float(rows[0].get("model_params_B", -1)),
            "n_tasks":    len(rows),
            "pearson_r":  round(float(r_val),  3),
            "p_value":    round(float(p_val),  4) if p_val is not None else None,
            "trend":      trend,
            "mean_gap_low":    float(np.mean([float(r["synergy_gap"]) for r in rows
                                              if int(float(r["task_complexity"])) == 0])) if any(int(float(r["task_complexity"])) == 0 for r in rows) else None,
            "mean_gap_medium": float(np.mean([float(r["synergy_gap"]) for r in rows
                                              if int(float(r["task_complexity"])) == 1])) if any(int(float(r["task_complexity"])) == 1 for r in rows) else None,
            "mean_gap_high":   float(np.mean([float(r["synergy_gap"]) for r in rows
                                              if int(float(r["task_complexity"])) == 2])) if any(int(float(r["task_complexity"])) == 2 for r in rows) else None,
        })

    # Architecture type averages
    arch_type_gaps: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        arch_type_gaps[r.get("arch_type", "unknown")][int(float(r["task_complexity"]))].append(
            float(r["synergy_gap"]))

    arch_summary = {
        at: {c: (float(np.mean(vs)), float(np.std(vs)))
             for c, vs in comp_dict.items()}
        for at, comp_dict in arch_type_gaps.items()
    }

    return {"model_stats": model_stats, "arch_summary": arch_summary}


def print_report(results: List[Dict], stats: Dict) -> str:
    lines = ["# Synergy Gap Experiment — Results Report", ""]

    # Table 1: per-model statistics
    lines += ["## Table 1: Per-Model Pearson Correlation (Task Complexity → Synergy Gap)", ""]
    lines += ["| Model | Family | Arch Type | r | p | n | Trend |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in sorted(stats["model_stats"], key=lambda x: x["family"]):
        p_str = f"{s['p_value']:.4f}" if s["p_value"] is not None else "—"
        lines.append(f"| {s['model_short']} | {s['family']} | {s['arch_type']} "
                     f"| {s['pearson_r']:+.3f} | {p_str} | {s['n_tasks']} "
                     f"| **{s['trend']}** |")
    lines.append("")

    # Trend summary
    trends = {}
    for s in stats["model_stats"]:
        trends.setdefault(s["trend"], []).append(s["model_short"])
    lines += ["## Trend Summary", ""]
    for trend, models in trends.items():
        lines.append(f"- **{trend} Gap** ({len(models)} models): "
                     + ", ".join(models))
    lines.append("")

    # Architecture type table
    lines += ["## Table 2: Mean Synergy Gap by Architecture Type and Complexity", ""]
    lines += ["| Architecture | Low | Medium | High |",
              "| --- | --- | --- | --- |"]
    for at, comp_dict in sorted(stats["arch_summary"].items()):
        def fmt(c):
            if c not in comp_dict:
                return "—"
            mu, sd = comp_dict[c]
            return f"{mu:.4f} ± {sd:.4f}"
        lines.append(f"| {at} | {fmt(0)} | {fmt(1)} | {fmt(2)} |")
    lines.append("")

    # Key finding
    closing_count = sum(1 for s in stats["model_stats"] if s["trend"] == "Closing")
    total = len(stats["model_stats"])
    lines += ["## Key Finding", ""]
    if total > 0:
        pct = 100 * closing_count / total
        lines.append(
            f"{closing_count}/{total} models ({pct:.0f}%) show a **Closing Gap** trend "
            f"(Pearson r < −0.3, p < 0.1), consistent with the hypothesis that "
            f"harder tasks force models to utilise more of the residual-path space.")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report


# =============================================================================
# Visualisation — Figure 1: Synergy Gap vs Complexity (main paper figure)
# =============================================================================

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
_JITTER  = 0.05


def _family_key(name: str) -> str:
    """Canonical family name used to group subplots."""
    m = name.lower()
    if "pythia"  in m: return "Pythia"
    if "gpt-j"   in m: return "GPT-J"
    if "gpt-neo" in m: return "GPT-Neo"
    if "gpt2"    in m: return "GPT-2"
    if "opt"     in m: return "OPT"
    if "llama"   in m: return "LLaMA"
    if "falcon"  in m: return "Falcon"
    return "Other"


def plot_gap_vs_complexity(results: List[Dict], output_path: str,
                           stats: Dict) -> None:
    """
    Main paper figure: one subplot per model family showing ΔH vs task complexity.
    """
    # Group models by family
    family_models: Dict[str, List[str]] = {}
    for r in results:
        fam = _family_key(r["model"])
        if fam not in family_models:
            family_models[fam] = []
        if r["model"] not in family_models[fam]:
            family_models[fam].append(r["model"])

    families = sorted(family_models.keys())
    n_fam    = len(families)
    if n_fam == 0:
        log.warning("No data for figure 1.")
        return

    ncols = min(n_fam, 3)
    nrows = (n_fam + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=False, constrained_layout=True)
    if n_fam == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(nrows, ncols)

    rng = np.random.default_rng(0)

    for fam_idx, family in enumerate(families):
        ax  = axes[fam_idx // ncols, fam_idx % ncols]
        models_in_fam = sorted(
            family_models[family],
            key=lambda m: float(MODEL_PARAMS_B.get(m, 0)))

        palette = (sns.color_palette("tab10", len(models_in_fam))
                   if _HAS_SEABORN
                   else plt.cm.tab10(np.linspace(0, 0.9, len(models_in_fam))))

        for i, model in enumerate(models_in_fam):
            model_rows = sorted(
                [r for r in results if r["model"] == model],
                key=lambda r: float(r["task_complexity"]))
            if not model_rows:
                continue
            xs = np.array([float(r["task_complexity"]) for r in model_rows])
            ys = np.array([float(r["synergy_gap"])     for r in model_rows])
            errs = np.array([float(r.get("empirical_entropy_std", 0)) for r in model_rows])

            color  = palette[i]
            marker = _MARKERS[i % len(_MARKERS)]
            jitter = rng.uniform(-_JITTER, _JITTER, len(xs))
            label  = _short_name(model)

            ax.plot(xs, ys, color=color, lw=1.5, alpha=0.8, zorder=2)
            ax.errorbar(xs + jitter, ys, yerr=errs,
                        fmt=marker, color=color, ms=7,
                        elinewidth=1, capsize=3, label=label,
                        zorder=3, markeredgecolor="white", markeredgewidth=0.5)

            # Inline Pearson r annotation
            ms = [s for s in stats["model_stats"] if s["model"] == model]
            if ms:
                ax.annotate(f"r={ms[0]['pearson_r']:+.2f}",
                            xy=(xs[-1], ys[-1]),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=7.5, color=color, va="center")

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5,
                   label="ΔH = 0 (full utilisation)")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_xlim(-0.4, 2.8)
        ax.set_xlabel("Task Complexity", fontsize=10)
        ax.set_ylabel(r"$\Delta H = H_{\rm ana} - H_{\rm emp}$  (bits)", fontsize=9)
        ax.set_title(family, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85,
                  borderpad=0.5, handlelength=1.4)

    # Hide unused subplots
    for idx in range(n_fam, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        r"Synergy Gap $\Delta H$ vs Task Complexity"
        "\n(error bars = std of empirical entropy across samples)",
        fontsize=12, fontweight="bold")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Fig 1 saved -> %s", output_path)
    plt.close(fig)


# =============================================================================
# Figure 2: Architecture type comparison (boxplot)
# =============================================================================

def plot_arch_comparison(results: List[Dict], output_path: str) -> None:
    """
    Box/strip plot: ΔH distribution for each (arch_type, complexity) combination.
    Shows how normalization (pre-LN vs post-LN) and parallelism affect the gap.
    """
    complexity_labels = ["Low", "Medium", "High"]
    arch_types = sorted({r.get("arch_type", "unknown") for r in results})
    if not arch_types:
        return

    arch_display = {
        "sequential-pre-LN":  "Sequential\nPre-LN",
        "sequential-post-LN": "Sequential\nPost-LN",
        "parallel":           "Parallel\n(ternary)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True,
                             constrained_layout=True)

    colors_arch = (sns.color_palette("Set2", len(arch_types))
                   if _HAS_SEABORN else plt.cm.Set2(np.linspace(0, 0.8, len(arch_types))))
    color_map = dict(zip(arch_types, colors_arch))

    for c_idx, (ax, c_label) in enumerate(zip(axes, complexity_labels)):
        for a_idx, at in enumerate(arch_types):
            vals = [float(r["synergy_gap"])
                    for r in results
                    if r.get("arch_type") == at
                    and int(float(r["task_complexity"])) == c_idx]
            if not vals:
                continue

            x     = a_idx
            color = color_map[at]

            # Box
            bp = ax.boxplot([vals], positions=[x], widths=0.45, patch_artist=True,
                            medianprops=dict(color="black", lw=2),
                            boxprops=dict(facecolor=(*color[:3], 0.6) if len(color) >= 3 else color),
                            whiskerprops=dict(color=color),
                            capprops=dict(color=color),
                            flierprops=dict(marker="o", color=color, alpha=0.5, ms=4))
            # Jittered strip overlay
            rng   = np.random.default_rng(c_idx * 10 + a_idx)
            jx    = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), x) + jx, vals,
                       color=color, alpha=0.7, s=18, zorder=3,
                       edgecolors="white", linewidths=0.3)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.45)
        ax.set_title(f"{c_label} Complexity", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(arch_types)))
        ax.set_xticklabels([arch_display.get(at, at) for at in arch_types], fontsize=9)
        if c_idx == 0:
            ax.set_ylabel(r"$\Delta H$  (bits)", fontsize=10)

    # Legend patches
    patches = [mpatches.Patch(color=color_map[at],
                               label=arch_display.get(at, at).replace("\n", " "))
               for at in arch_types]
    fig.legend(handles=patches, loc="upper center", ncol=len(arch_types),
               fontsize=9, bbox_to_anchor=(0.5, 1.05), framealpha=0.9)

    fig.suptitle("Synergy Gap by Architecture Type at Each Complexity Level",
                 fontsize=12, fontweight="bold", y=1.10)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Fig 2 saved -> %s", output_path)
    plt.close(fig)


# =============================================================================
# Figure 3: Scaling curves (log params vs ΔH)
# =============================================================================

def plot_scaling(results: List[Dict], output_path: str) -> None:
    """
    Log(model size in B params) on X, Synergy Gap on Y.
    One curve per complexity level; marker shape encodes architecture family.
    """
    valid = [r for r in results
             if float(r.get("model_params_B", -1)) > 0]
    if not valid:
        log.warning("No model_params_B data — skipping figure 3.")
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    complexity_colors = {0: "#2196F3", 1: "#FF9800", 2: "#E53935"}
    complexity_names  = {0: "Low", 1: "Medium", 2: "High"}
    family_markers    = {"Pythia": "o", "GPT-2": "s", "GPT-J": "^",
                         "GPT-Neo": "D", "OPT": "v", "LLaMA": "P",
                         "Other": "X"}

    plotted: set = set()
    rng = np.random.default_rng(1)

    for r in valid:
        c      = int(float(r["task_complexity"]))
        fam    = _family_key(r["model"])
        params = float(r["model_params_B"])
        gap    = float(r["synergy_gap"])
        color  = complexity_colors.get(c, "gray")
        marker = family_markers.get(fam, "X")

        jitter = rng.uniform(-0.02, 0.02)
        ax.scatter(np.log10(params) + jitter, gap,
                   color=color, marker=marker, s=55, alpha=0.8, zorder=3,
                   edgecolors="white", linewidths=0.4)

    # Trend lines per complexity level
    for c in sorted(complexity_colors):
        rows_c = [r for r in valid if int(float(r["task_complexity"])) == c]
        if len(rows_c) < 3:
            continue
        xs_log = np.log10([float(r["model_params_B"]) for r in rows_c])
        ys     = np.array([float(r["synergy_gap"])    for r in rows_c])
        coef   = np.polyfit(xs_log, ys, 1)
        xs_fit = np.linspace(xs_log.min(), xs_log.max(), 80)
        ax.plot(xs_fit, np.polyval(coef, xs_fit),
                color=complexity_colors[c], lw=2, alpha=0.6, ls="--",
                label=f"{complexity_names[c]} (slope={coef[0]:+.3f})")

    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.4)
    ax.set_xlabel("Model Size  [log10 B params]", fontsize=11)
    ax.set_ylabel(r"$\Delta H = H_{\rm ana} - H_{\rm emp}$  (bits)", fontsize=11)
    ax.set_title("Synergy Gap vs Model Scale, by Task Complexity", fontsize=12,
                 fontweight="bold")

    # Complexity legend (color)
    complexity_patches = [mpatches.Patch(color=v, label=f"{complexity_names[k]} complexity")
                          for k, v in complexity_colors.items()]
    # Family legend (marker shape)
    family_handles = [plt.Line2D([0], [0], marker=mv, color="gray", ls="none",
                                  ms=7, label=fk)
                      for fk, mv in family_markers.items()
                      if any(_family_key(r["model"]) == fk for r in valid)]
    leg1 = ax.legend(handles=complexity_patches, loc="upper left",
                     title="Complexity", fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=family_handles, loc="lower right",
              title="Family", fontsize=8, title_fontsize=9)

    # X-tick labels: actual sizes
    ticks = np.arange(
        np.floor(np.log10(min(float(r["model_params_B"]) for r in valid))),
        np.ceil( np.log10(max(float(r["model_params_B"]) for r in valid))) + 0.5,
        0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{10**t:.3g}B" for t in ticks], fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Fig 3 saved -> %s", output_path)
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data source
    parser.add_argument("--models", default=None,
        help="Comma-separated HuggingFace model IDs")
    parser.add_argument("--model_group", default="small",
        choices=list(MODEL_GROUPS.keys()),
        help=f"Predefined group (default: small).  Options: {list(MODEL_GROUPS.keys())}")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names from: " + ", ".join(TASKS.keys()))

    # Experiment settings
    parser.add_argument("--n_samples",        type=int,   default=50)
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--epsilon_quantile", type=float, default=0.25)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output_dir",       default="results")

    # Resume / plot only
    parser.add_argument("--resume", default=None,
        help="Path to existing CSV to resume from")
    parser.add_argument("--plot_only", default=None,
        help="Skip experiment; load this CSV and regenerate figures only")

    args = parser.parse_args()

    output_dir   = args.output_dir
    csv_path     = os.path.join(output_dir, "synergy_gap_results.csv")
    fig1_path    = os.path.join(output_dir, "fig1_gap_vs_complexity.png")
    fig2_path    = os.path.join(output_dir, "fig2_arch_comparison.png")
    fig3_path    = os.path.join(output_dir, "fig3_scaling.png")
    report_path  = os.path.join(output_dir, "synergy_gap_report.md")

    # ── Load existing data or run experiment ──────────────────────────────────
    if args.plot_only:
        log.info("--plot_only: loading %s", args.plot_only)
        results = load_csv(args.plot_only)
    else:
        if args.models:
            model_names = [m.strip() for m in args.models.split(",")]
        else:
            model_names = MODEL_GROUPS[args.model_group]

        task_names = [t.strip() for t in args.tasks.split(",")]
        unknown = [t for t in task_names if t not in TASKS]
        if unknown:
            parser.error(f"Unknown tasks: {unknown}. "
                         f"Valid: {list(TASKS.keys())}")

        log.info("Models (%d): %s", len(model_names), model_names)
        log.info("Tasks  (%d): %s", len(task_names),  task_names)
        log.info("Samples/task: %d  |  device: %s  |  ε-quantile: %.2f",
                 args.n_samples, args.device, args.epsilon_quantile)

        results = run_experiment(
            model_names      = model_names,
            task_names       = task_names,
            n_samples        = args.n_samples,
            output_dir       = output_dir,
            device           = args.device,
            epsilon_quantile = args.epsilon_quantile,
            seed             = args.seed,
            resume_csv       = args.resume,
        )
        _save_csv(results, csv_path)
        log.info("CSV saved -> %s", csv_path)

    if not results:
        log.error("No results to analyse.")
        sys.exit(1)

    # ── Analysis ──────────────────────────────────────────────────────────────
    stats  = analyse(results)
    report = print_report(results, stats)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved -> %s", report_path)

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_gap_vs_complexity(results, fig1_path, stats)
    plot_arch_comparison(results,   fig2_path)
    plot_scaling(results,           fig3_path)

    log.info("")
    log.info("All outputs written to %s/", output_dir)
    log.info("  CSV    : %s", csv_path)
    log.info("  Fig 1  : %s", fig1_path)
    log.info("  Fig 2  : %s", fig2_path)
    log.info("  Fig 3  : %s", fig3_path)
    log.info("  Report : %s", report_path)


if __name__ == "__main__":
    main()
