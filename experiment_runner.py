#!/usr/bin/env python3
"""
experiment_runner.py — Path Distribution Theory: Experimental Loop
====================================================================

For each (model, task):
  1. Analytical Path Entropy  H_ana   (architecture only)
  2. Empirical  Path Entropy  H_emp   (AtP active subgraph)
  3. Synergy Gap              ΔH = H_ana − H_emp
  4. Tail-Mass Ratio          τ_k
  5. Task Accuracy            (log-prob comparison for classification / MCQ)

Outputs
  results/path_metrics.csv
  results/entropy_vs_complexity.png

Usage
-----
  # Quick test — GPT-2 on CPU, 5 samples:
  python experiment_runner.py --models gpt2 --n_samples 5 --device cpu

  # Pythia scaling series:
  python experiment_runner.py --model_group pythia --tasks sst2,boolq,arc_easy,hellaswag

  # GPT-2 family:
  python experiment_runner.py --model_group gpt2 --tasks sst2,boolq,piqa,arc_easy

  # GPT-J + Pythia comparison:
  python experiment_runner.py \\
      --models "EleutherAI/gpt-j-6b,EleutherAI/pythia-6.9b" \\
      --tasks sst2,boolq,arc_easy,arc_challenge,hellaswag,gsm8k

  # Full A100 sweep:
  python experiment_runner.py --model_group all --n_samples 50
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
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
# Model catalogue
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace authentication helper
# ─────────────────────────────────────────────────────────────────────────────

def _hf_login(token: Optional[str] = None) -> None:
    """
    Authenticate with HuggingFace Hub for the current Python session.

    Token resolution order:
      1. `token` argument (from --hf_token CLI flag)
      2. HF_TOKEN  environment variable
      3. HUGGING_FACE_HUB_TOKEN  environment variable
      4. Cached credentials from `huggingface-cli login`  (automatic — no action needed)

    If no token is found, a warning is logged.  Public / non-gated models
    work without authentication.  Gated models (meta-llama/*) require a token.
    """
    import os
    resolved = (token
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if resolved:
        try:
            import huggingface_hub
            huggingface_hub.login(token=resolved, add_to_git_credential=False)
            log.info("HuggingFace: authenticated via token.")
        except Exception as exc:
            log.warning("HuggingFace login failed: %s", exc)
    else:
        log.debug("No HF_TOKEN found — using cached credentials or public access only.")


MODEL_GROUPS: Dict[str, List[str]] = {
    # Pythia family — identical architecture, vary only in scale (sequential, pre-RMSNorm)
    "pythia": [
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b",
    ],
    # GPT-2 family — sequential pre-LN
    "gpt2": [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
    ],
    # GPT-Neo family — sequential (GPT-Neo) + parallel (GPT-J)
    "neo": [
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/gpt-j-6b",      # parallel architecture
    ],
    # Llama-3 — sequential, Pre-RMSNorm + RoPE + SwiGLU (requires 4-bit for 70B)
    # NousResearch mirrors are NOT gated; meta-llama/* require HF token + Meta approval.
    "llama": [
        "NousResearch/Meta-Llama-3-8B",        # non-gated community mirror
        "NousResearch/Meta-Llama-3-70B",       # non-gated community mirror (~38 GB 4-bit)
    ],
    # Gated originals — require HF token + Meta approval (huggingface.co/meta-llama)
    "llama_gated": [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B",
    ],
    # Instruct variants — community mirrors
    "llama_instruct": [
        "NousResearch/Meta-Llama-3-8B-Instruct",
        "NousResearch/Meta-Llama-3-70B-Instruct",
    ],
    # Large cross-architecture comparison (all ≥6B, 4-bit recommended)
    "large": [
        "NousResearch/Meta-Llama-3-8B",
        "EleutherAI/gpt-j-6b",
        "EleutherAI/pythia-6.9b",
    ],
}
MODEL_GROUPS["all"] = (
    MODEL_GROUPS["pythia"]
    + MODEL_GROUPS["gpt2"]
    + MODEL_GROUPS["neo"]
    + MODEL_GROUPS["llama"]
)

# Approximate parameter counts for plot annotations
MODEL_PARAMS: Dict[str, float] = {
    "EleutherAI/pythia-70m":                    0.07,
    "EleutherAI/pythia-160m":                   0.16,
    "EleutherAI/pythia-410m":                   0.41,
    "EleutherAI/pythia-1b":                     1.0,
    "EleutherAI/pythia-2.8b":                   2.8,
    "EleutherAI/pythia-6.9b":                   6.9,
    "gpt2":                                     0.12,
    "gpt2-medium":                              0.35,
    "gpt2-large":                               0.77,
    "gpt2-xl":                                  1.5,
    "EleutherAI/gpt-neo-125m":                  0.125,
    "EleutherAI/gpt-neo-1.3B":                  1.3,
    "EleutherAI/gpt-neo-2.7B":                  2.7,
    "EleutherAI/gpt-j-6b":                      6.0,
    # Llama-3 — gated originals
    "meta-llama/Meta-Llama-3-8B":               8.0,
    "meta-llama/Meta-Llama-3-8B-Instruct":      8.0,
    "meta-llama/Meta-Llama-3-70B":              70.0,
    "meta-llama/Meta-Llama-3-70B-Instruct":    70.0,
    # Llama-3 — NousResearch non-gated mirrors (identical weights)
    "NousResearch/Meta-Llama-3-8B":             8.0,
    "NousResearch/Meta-Llama-3-8B-Instruct":    8.0,
    "NousResearch/Meta-Llama-3-70B":            70.0,
    "NousResearch/Meta-Llama-3-70B-Instruct":  70.0,
    "tiiuae/falcon-7b":                         7.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task catalogue
# ─────────────────────────────────────────────────────────────────────────────
# complexity: 0=Low  1=Medium  2=High
# eval_type:  "binary"    → compare two label tokens by max-logit
#             "choice"    → score each completion by mean log-prob
#             "generative"→ greedy-decode and regex-match answer

TASKS: Dict[str, Dict[str, Any]] = {
    # ── Low complexity ──────────────────────────────────────────────────────
    "sst2": {
        "hf_name": "sst2", "hf_split": "validation",
        "complexity": 0, "eval_type": "binary",
        "labels": [["negative", "Negative"], ["positive", "Positive"]],
    },
    "piqa": {
        "hf_name": "piqa", "hf_split": "validation",
        "complexity": 0, "eval_type": "choice",
    },
    # ── Medium complexity ────────────────────────────────────────────────────
    "boolq": {
        "hf_name": "boolq", "hf_split": "validation",
        "complexity": 1, "eval_type": "binary",
        "labels": [["no", "No", "false", "False"], ["yes", "Yes", "true", "True"]],
    },
    "arc_easy": {
        "hf_name": "ai2_arc", "hf_split": "validation", "hf_config": "ARC-Easy",
        "complexity": 1, "eval_type": "choice",
    },
    "winogrande": {
        "hf_name": "winogrande", "hf_split": "validation", "hf_config": "winogrande_xl",
        "complexity": 1, "eval_type": "choice",
    },
    # ── High complexity ──────────────────────────────────────────────────────
    "hellaswag": {
        "hf_name": "hellaswag", "hf_split": "validation",
        "complexity": 2, "eval_type": "choice",
    },
    "arc_challenge": {
        "hf_name": "ai2_arc", "hf_split": "validation", "hf_config": "ARC-Challenge",
        "complexity": 2, "eval_type": "choice",
    },
    "gsm8k": {
        "hf_name": "gsm8k", "hf_split": "test", "hf_config": "main",
        "complexity": 2, "eval_type": "generative",
    },
}

COMPLEXITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}
COMPLEXITY_TASK_LABELS = {
    0: "Low\n(SST-2 / PIQA)",
    1: "Medium\n(BoolQ / ARC-Easy)",
    2: "High\n(HellaSwag / GSM8K)",
}

DEFAULT_TASKS = ["sst2", "boolq", "arc_easy", "hellaswag", "gsm8k"]

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _is_llama_family(model_name: str) -> bool:
    """True for RoPE + RMSNorm models that need special TransformerLens kwargs."""
    return any(k in model_name.lower()
               for k in ("llama", "mistral", "gemma", "qwen", "phi"))


def load_model(
    model_name: str,
    device:     str = "cuda",
    quant:      str = "4bit",   # "4bit" | "8bit" | "none"
    hf_token:   Optional[str] = None,
):
    """
    Load a HookedTransformer with optional BitsAndBytes quantisation.

    Priority:
      "4bit" → NF4 double-quant (bf16 compute) → 8-bit int8 → bf16/fp16 native
      "8bit" → 8-bit int8 → bf16/fp16 native
      "none" → bf16 (Llama) or fp16 (others)

    Authentication:
      Pass hf_token for gated models (meta-llama/*).
      Non-gated mirrors (NousResearch/*) work without a token.
      Also respects HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars.

    Llama / RoPE-based models automatically receive the required TL kwargs:
      fold_ln=False, center_writing_weights=False, center_unembed=False
    """
    from transformer_lens import HookedTransformer

    is_llama  = _is_llama_family(model_name)
    nat_dtype = torch.bfloat16 if is_llama else torch.float16
    tl_kwargs = dict(fold_ln=False,
                     center_writing_weights=False,
                     center_unembed=False) if is_llama else {}

    log.info("Loading  %s  [quant=%s, dtype=%s]",
             model_name, quant, "bf16" if is_llama else "fp16")

    # Resolve token: explicit arg → env var
    import os
    token = (hf_token
             or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    tok_kw = {"token": token} if token else {}

    # ── 4-bit NF4 (QLoRA-style double quantisation) ───────────────────────
    if quant == "4bit":
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit               = True,
                bnb_4bit_quant_type        = "nf4",
                bnb_4bit_compute_dtype     = torch.bfloat16,
                bnb_4bit_use_double_quant  = True,
            )
            log.info("  → 4-bit NF4 …")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config = bnb_cfg,
                device_map          = "auto",
                torch_dtype         = torch.bfloat16,
                **tok_kw,
            )
            model = HookedTransformer.from_pretrained(
                model_name,
                hf_model       = hf_model,
                dtype          = torch.bfloat16,
                move_to_device = False,
                **tl_kwargs,
                **tok_kw,
            )
            model.eval()
            log.info("  ✓ 4-bit NF4 OK")
            return model
        except Exception as exc:
            log.warning("  4-bit failed (%s) — trying 8-bit …", exc)

    # ── 8-bit LLM.int8 ───────────────────────────────────────────────────
    if quant in ("4bit", "8bit"):
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_8bit              = True,
                llm_int8_threshold        = 6.0,
                llm_int8_has_fp16_weight  = False,
            )
            log.info("  → 8-bit int8 …")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config = bnb_cfg,
                device_map          = "auto",
                torch_dtype         = torch.float16,
                **tok_kw,
            )
            model = HookedTransformer.from_pretrained(
                model_name,
                hf_model       = hf_model,
                dtype          = torch.float16,
                move_to_device = False,
                **tl_kwargs,
                **tok_kw,
            )
            model.eval()
            log.info("  ✓ 8-bit OK")
            return model
        except Exception as exc:
            log.warning("  8-bit failed (%s) — native dtype fallback …", exc)

    # ── Native dtype (bf16 for Llama, fp16 otherwise) ────────────────────
    log.info("  → %s no-quant …", "bf16" if is_llama else "fp16")
    model = HookedTransformer.from_pretrained(
        model_name, dtype=nat_dtype, device=device, **tl_kwargs, **tok_kw)
    model.eval()
    log.info("  ✓ %s OK", "bf16" if is_llama else "fp16")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_task_samples(task_name: str, n_samples: int = 50, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset as hf_load

    cfg    = TASKS[task_name]
    kwargs: Dict[str, Any] = {"path": cfg["hf_name"], "split": cfg["hf_split"]}
    if "hf_config" in cfg:
        kwargs["name"] = cfg["hf_config"]

    ds      = hf_load(**kwargs)
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    samples = [s for idx in indices.tolist()
               if (s := _format_sample(task_name, ds[int(idx)])) is not None]
    log.info("  Loaded %d / %d samples for %s", len(samples), n_samples, task_name)
    return samples


def _format_sample(task_name: str, row: Dict) -> Optional[Dict]:
    """Convert a dataset row into a standardised sample dict."""

    if task_name == "sst2":
        return {
            "prompt": f"Review: {row['sentence']}\nSentiment (positive/negative):",
            "label": "positive" if row["label"] == 1 else "negative",
            "correct_class": int(row["label"]),
            "task": task_name,
        }

    if task_name == "boolq":
        return {
            "prompt": (
                f"Passage: {row['passage'][:400]}\n"
                f"Question: {row['question']}\nAnswer (yes/no):"
            ),
            "label": "yes" if row["answer"] else "no",
            "correct_class": int(row["answer"]),
            "task": task_name,
        }

    if task_name == "piqa":
        return {
            "prompt": f"Goal: {row['goal']}\nSolution:",
            "choices": [f" {row['sol1']}", f" {row['sol2']}"],
            "correct_class": int(row["label"]),
            "task": task_name,
        }

    if task_name in ("arc_easy", "arc_challenge"):
        choices_text = row["choices"]["text"]
        choices_label = row["choices"]["label"]
        correct_idx = choices_label.index(row["answerKey"]) if row["answerKey"] in choices_label else 0
        return {
            "prompt": f"Question: {row['question']}\nAnswer:",
            "choices": [f" {c}" for c in choices_text],
            "correct_class": correct_idx,
            "task": task_name,
        }

    if task_name == "hellaswag":
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        return {
            "prompt": ctx,
            "choices": [" " + e for e in row["endings"]],
            "correct_class": int(row["label"]),
            "task": task_name,
        }

    if task_name == "winogrande":
        # Two options fill the blank; evaluate as continuations
        blank_pos = row["sentence"].find("_")
        context   = row["sentence"][:blank_pos]
        return {
            "prompt": context,
            "choices": [row["option1"], row["option2"]],
            "correct_class": int(row["answer"]) - 1,   # "1"/"2" → 0/1
            "task": task_name,
        }

    if task_name == "gsm8k":
        answer_str = row.get("answer", "")
        match      = re.search(r"####\s*([\d,.\-]+)", answer_str)
        label      = match.group(1).replace(",", "") if match else ""
        return {
            "prompt": f"Problem: {row['question']}\nSolution:",
            "label": label,
            "correct_class": None,
            "task": task_name,
        }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy  (log-probability based — robust across tokenisers)
# ─────────────────────────────────────────────────────────────────────────────

def _label_token_ids(model, words: List[str]) -> List[int]:
    """
    Return all single-token IDs corresponding to any variant in `words`.
    Includes leading-space variants since most LLMs tokenise mid-sentence
    words with a space prefix.
    """
    variants: List[str] = []
    for w in words:
        variants += [w, w.capitalize(), w.upper(), f" {w}", f" {w.capitalize()}"]
    ids: List[int] = []
    for v in variants:
        enc = model.tokenizer.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return list(set(ids))


def _score_continuation(model, prompt_tokens: torch.Tensor, cont_tokens: torch.Tensor) -> float:
    """
    Mean per-token log-probability of `cont_tokens` given `prompt_tokens`.
    Used for multiple-choice / completion scoring.
    """
    if cont_tokens.shape[-1] == 0:
        return -float("inf")
    full   = torch.cat([prompt_tokens, cont_tokens], dim=-1)
    with torch.no_grad():
        logits = model(full, return_type="logits")        # [1, seq, vocab]
    log_probs = torch.log_softmax(logits[0], dim=-1)
    n_prompt  = prompt_tokens.shape[-1]
    n_cont    = cont_tokens.shape[-1]
    score = 0.0
    for i in range(n_cont):
        pos     = n_prompt + i - 1          # logit at position predicts token at pos+1
        tok_id  = cont_tokens[0, i]
        score  += log_probs[pos, tok_id].item()
    return score / n_cont


def check_accuracy(task_name: str, model, tokens: torch.Tensor, sample: Dict) -> bool:
    """
    Compute task accuracy using log-probability comparison.

    Binary tasks  → compare max-logit of positive-class tokens vs negative-class tokens.
    Choice tasks  → score each continuation; pick argmax.
    Generative    → decode first 20 tokens, extract numeric answer.
    """
    eval_type = TASKS[task_name]["eval_type"]

    # ── Binary (SST-2, BoolQ) ──────────────────────────────────────────────
    if eval_type == "binary":
        label_groups = TASKS[task_name]["labels"]   # [[neg words], [pos words]]
        with torch.no_grad():
            logits = model(tokens, return_type="logits")
        last = logits[0, -1]    # [vocab]
        scores = []
        for words in label_groups:
            ids = _label_token_ids(model, words)
            if ids:
                scores.append(float(last[ids].max()))
            else:
                scores.append(-float("inf"))
        pred_class = int(np.argmax(scores))
        return pred_class == sample["correct_class"]

    # ── Multiple-choice / completion (PIQA, ARC, HellaSwag, WinoGrande) ──
    if eval_type == "choice":
        choices = sample.get("choices", [])
        if not choices:
            return False
        scores = []
        for choice in choices:
            cont_ids   = model.tokenizer.encode(choice, add_special_tokens=False)
            cont_tok   = torch.tensor([cont_ids], dtype=torch.long, device=tokens.device)
            scores.append(_score_continuation(model, tokens, cont_tok))
        pred_class = int(np.argmax(scores))
        return pred_class == sample["correct_class"]

    # ── Generative (GSM8K) ────────────────────────────────────────────────
    if eval_type == "generative":
        expected = sample.get("label", "")
        if not expected:
            return False
        with torch.no_grad():
            out = model.generate(tokens, max_new_tokens=20, do_sample=False)
        new_tokens = out[0, tokens.shape[-1]:]
        decoded    = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        nums = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", decoded)
        if nums:
            num_str = nums[0].replace(",", "")
            try:
                return abs(float(num_str) - float(expected)) < 0.5
            except ValueError:
                pass
        return False

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model_names:   List[str],
    task_names:    List[str],
    n_samples:     int   = 50,
    output_dir:    str   = "results",
    device:        str   = "cuda",
    mass_coverage: float = 0.90,
    quant:         str   = "4bit",
    max_seq_len:   int   = 512,
    hf_token:      Optional[str] = None,
    seed:          int   = 42,
) -> List[Dict[str, Any]]:
    """
    Single-pass experiment loop with per-sample mass-coverage subgraph selection.

    For each sample the active subgraph is chosen so that the minimum number
    of edges covering >= mass_coverage of the total attribution mass are kept.
    """
    from path_analyzer import (PathAnalyzer, _to_metrics,
                                select_active_edges_by_mass_coverage)

    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for model_name in model_names:
        try:
            model = load_model(model_name, device=device,
                               quant=quant, hf_token=hf_token)
        except Exception as exc:
            log.error("Could not load %s: %s", model_name, exc)
            continue

        analyzer  = PathAnalyzer(model)
        arch_info = analyzer.architecture_summary()
        log.info("Architecture: %s", arch_info)

        ana_metrics = analyzer.analytical_path_distribution()
        log.info(
            "  Analytical  H=%.4f bits  E[L]=%.2f  τ=%.4f  max_l=%d",
            ana_metrics.entropy, ana_metrics.mean_path_length,
            ana_metrics.tail_mass_ratio, ana_metrics.max_path_len,
        )

        for task_name in task_names:
            if task_name not in TASKS:
                log.warning("Unknown task '%s' — skipping.", task_name)
                continue
            log.info("")
            log.info("─── Model=%-35s  Task=%s ───", model_name, task_name)

            try:
                samples = load_task_samples(task_name, n_samples=n_samples, seed=seed)
            except Exception as exc:
                log.error("  Could not load task %s: %s", task_name, exc)
                continue

            # ── Single-pass: AtP + accuracy + per-sample path metrics ──────
            emp_H:        List[float] = []
            emp_mu:       List[float] = []
            emp_tau:      List[float] = []
            emp_n_active: List[int]   = []
            emp_epsilon:  List[float] = []
            per_corr:     List[bool]  = []

            for i, sample in enumerate(samples):
                try:
                    tokens = model.to_tokens(
                        sample["prompt"], prepend_bos=True
                    ).to(device)
                    if tokens.shape[-1] > max_seq_len:
                        tokens = tokens[:, -max_seq_len:]

                    per_corr.append(check_accuracy(task_name, model, tokens, sample))

                    a_sc_t, m_sc_t = analyzer.compute_attribution_scores(tokens)
                    a_sc = a_sc_t.cpu().float().numpy()
                    m_sc = m_sc_t.cpu().float().numpy()

                    # Per-sample mass-coverage active subgraph
                    act_a, act_m, epsilon_s, k = \
                        select_active_edges_by_mass_coverage(
                            a_sc, m_sc, mass_coverage)

                    m_obj = _to_metrics(analyzer._path_count_dp(act_a, act_m))
                    emp_H.append(m_obj.entropy)
                    emp_mu.append(m_obj.mean_path_length)
                    emp_tau.append(m_obj.tail_mass_ratio)
                    emp_n_active.append(k)
                    emp_epsilon.append(epsilon_s)

                    if (i + 1) % 10 == 0:
                        log.info("  … %d/%d samples  (last k=%d, ε=%.2e)",
                                 i + 1, len(samples), k, epsilon_s)

                except Exception as exc:
                    log.warning("  Sample %d failed: %s", i, exc)

            if not emp_H:
                log.warning("  No valid samples for %s — skipping.", task_name)
                continue

            accuracy   = float(np.mean(per_corr))
            mean_H     = float(np.mean(emp_H))
            std_H      = float(np.std(emp_H))  if len(emp_H) > 1 else 0.0
            mean_mu    = float(np.mean(emp_mu))
            finite     = [t for t in emp_tau if np.isfinite(t)]
            mean_tau   = float(np.mean(finite)) if finite else float("inf")
            mean_n_act = float(np.mean(emp_n_active))
            mean_eps   = float(np.mean(emp_epsilon))

            result: Dict[str, Any] = {
                "model":                   model_name,
                "model_params_B":          MODEL_PARAMS.get(model_name, -1),
                "task":                    task_name,
                "task_complexity":         TASKS[task_name]["complexity"],
                "accuracy":                round(accuracy,  4),
                # ── analytical ──
                "analytical_entropy":      round(ana_metrics.entropy,          4),
                "analytical_mean_path":    round(ana_metrics.mean_path_length, 4),
                "analytical_tail_ratio":   round(ana_metrics.tail_mass_ratio,  4),
                # ── empirical ──
                "empirical_entropy":       round(mean_H,    4),
                "empirical_entropy_std":   round(std_H,     4),
                "empirical_mean_path":     round(mean_mu,   4),
                "empirical_tail_ratio":    round(mean_tau,  4),
                "mean_active_edges":       round(mean_n_act, 2),
                # ── derived ──
                "synergy_gap":             round(ana_metrics.entropy - mean_H, 4),
                # ── meta ──
                "n_layers":                arch_info["n_layers"],
                "architecture":            arch_info["architecture"],
                "mean_epsilon":            round(mean_eps, 8),
                "mass_coverage":           mass_coverage,
                "n_samples_used":          len(emp_H),
                "eval_type":               TASKS[task_name]["eval_type"],
            }
            results.append(result)

            log.info(
                "  Acc=%.3f  H_ana=%.3f  H_emp=%.3f±%.3f  Gap=%.3f  "
                "E[L]=%.2f  k=%.1f  ε=%.2e",
                accuracy, ana_metrics.entropy, mean_H, std_H,
                result["synergy_gap"], mean_mu, mean_n_act, mean_eps,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results: List[Dict], path: str) -> None:
    if not results:
        log.warning("No results — CSV not written.")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info("Results → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation  (4-panel figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: List[Dict], output_path: str = "results/entropy_vs_complexity.png") -> None:
    if not results:
        log.warning("No results — plot skipped.")
        return

    models       = sorted({r["model"] for r in results})
    complexities = sorted({r["task_complexity"] for r in results})

    cmap    = plt.get_cmap("tab10")
    colors  = {m: cmap(i % 10) for i, m in enumerate(models)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "h"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Path Distribution Theory — Empirical Results", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    def _means(rows, key):
        xs, ys, es = [], [], []
        for c in complexities:
            vals = [r[key] for r in rows if r["task_complexity"] == c]
            if vals:
                xs.append(c)
                ys.append(float(np.mean(vals)))
                es.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        return xs, ys, es

    xlabels = [COMPLEXITY_TASK_LABELS.get(c, str(c)) for c in complexities]

    for idx, mname in enumerate(models):
        rows  = [r for r in results if r["model"] == mname]
        label = mname.split("/")[-1]
        mk    = markers[idx % len(markers)]
        col   = colors[mname]

        # Panel 1 — Empirical entropy
        xs, ys, es = _means(rows, "empirical_entropy")
        ax1.errorbar(xs, ys, yerr=es, marker=mk, color=col,
                     label=label, capsize=4, linewidth=2, markersize=8)

        # Panel 2 — Synergy Gap
        xs, ys, _ = _means(rows, "synergy_gap")
        ax2.plot(xs, ys, marker=mk, color=col, label=label, linewidth=2, markersize=8)

        # Panel 3 — Mean path length
        xs, ys, _ = _means(rows, "empirical_mean_path")
        ax3.plot(xs, ys, marker=mk, color=col, label=label, linewidth=2, markersize=8)
        if rows:
            ana_mu = rows[0]["analytical_mean_path"]
            ax3.axhline(ana_mu, color=col, linestyle=":", linewidth=1, alpha=0.5)

        # Panel 4 — Accuracy
        xs, ys, _ = _means(rows, "accuracy")
        ax4.plot(xs, ys, marker=mk, color=col, label=label, linewidth=2, markersize=8)

    for ax, title, ylabel in [
        (ax1, "Empirical Path Entropy",       "H(π̂)  [bits]"),
        (ax2, "Synergy Gap  ΔH = H(π)−H(π̂)", "ΔH  [bits]"),
        (ax3, "Mean Path Length  (dotted=analytical)", "E[L]"),
        (ax4, "Task Accuracy",                "Accuracy"),
    ]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Task Complexity", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(complexities)
        ax.set_xticklabels(xlabels, fontsize=9, ha="center")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    ax2.axhline(0, color="grey", linestyle="--", linewidth=1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Plot → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Scaling plot  (H_emp vs model size, per task complexity)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scaling(results: List[Dict], output_path: str = "results/scaling_curves.png") -> None:
    """Entropy and Synergy Gap vs log(model size) — one panel per complexity level."""
    rows_with_size = [r for r in results if r.get("model_params_B", -1) > 0]
    if not rows_with_size:
        return

    complexities = sorted({r["task_complexity"] for r in rows_with_size})
    cmap   = plt.get_cmap("Set1")
    colors = {c: cmap(i) for i, c in enumerate(complexities)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Scaling Behaviour of Path Entropy Metrics", fontsize=13, fontweight="bold")

    for c in complexities:
        c_rows = sorted(
            [r for r in rows_with_size if r["task_complexity"] == c],
            key=lambda r: r["model_params_B"],
        )
        if not c_rows:
            continue
        sizes  = [r["model_params_B"] for r in c_rows]
        h_emp  = [r["empirical_entropy"] for r in c_rows]
        h_gap  = [r["synergy_gap"]       for r in c_rows]
        col    = colors[c]
        label  = COMPLEXITY_LABELS[c]

        ax1.plot(sizes, h_emp, "o-", color=col, label=label, linewidth=2, markersize=7)
        ax2.plot(sizes, h_gap, "s--", color=col, label=label, linewidth=2, markersize=7)

    for ax, title, ylabel in [
        (ax1, "Empirical Entropy vs Model Size", "H(π̂)  [bits]"),
        (ax2, "Synergy Gap vs Model Size",        "ΔH  [bits]"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("Model size  [B params]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Scaling plot → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]) -> None:
    if not results:
        return
    w = 28
    hdr = (f"{'Model':{w}} {'Task':14} {'Acc':6} "
           f"{'H_ana':7} {'H_emp':7} {'Gap':7} {'E[L]':6} {'τ':7}")
    bar = "=" * len(hdr)
    print()
    print(bar)
    print("  PATH DISTRIBUTION THEORY — RESULTS SUMMARY")
    print(bar)
    print(hdr)
    print("─" * len(hdr))

    # group by complexity so the table reads Low→Medium→High
    for c in sorted({r["task_complexity"] for r in results}):
        c_rows = [r for r in results if r["task_complexity"] == c]
        if c_rows:
            print(f"  ── {COMPLEXITY_LABELS[c]} complexity {'─'*40}")
        for r in c_rows:
            tau_str = f"{r['empirical_tail_ratio']:.4f}" if np.isfinite(r["empirical_tail_ratio"]) else "  inf "
            print(
                f"  {r['model'].split('/')[-1]:{w}} "
                f"{r['task']:14} "
                f"{r['accuracy']:6.3f} "
                f"{r['analytical_entropy']:7.3f} "
                f"{r['empirical_entropy']:7.3f} "
                f"{r['synergy_gap']:7.3f} "
                f"{r['empirical_mean_path']:6.2f} "
                f"{tau_str}"
            )
    print(bar)

    # Correlation summary: does Synergy Gap increase with complexity?
    if len(results) >= 3:
        complexities = [r["task_complexity"] for r in results]
        gaps         = [r["synergy_gap"]       for r in results]
        corr = float(np.corrcoef(complexities, gaps)[0, 1]) if len(set(complexities)) > 1 else 0.0
        print(f"\n  Pearson r(complexity, ΔH) = {corr:+.3f}  "
              f"({'↑ gap grows with complexity' if corr > 0.1 else '↓ gap shrinks' if corr < -0.1 else '— no clear trend'})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Path Distribution Theory — mechanistic interpretability pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model groups (--model_group):
  pythia         EleutherAI/pythia-70m … pythia-6.9b  (6 scales, sequential Pre-RMSNorm)
  gpt2           gpt2 … gpt2-xl                       (4 scales, sequential Pre-LN)
  neo            gpt-neo-125m … gpt-j-6b              (3 sequential + 1 parallel)
  llama          Meta-Llama-3-8B, Meta-Llama-3-70B    (sequential Pre-RMSNorm, RoPE)
  llama_instruct Meta-Llama-3-8B-Instruct, 70B-Instruct
  large          Llama-3-8B, gpt-j-6b, pythia-6.9b
  all            pythia + gpt2 + neo + llama

Tasks available:
  Low:    sst2, piqa
  Medium: boolq, arc_easy, winogrande
  High:   hellaswag, arc_challenge, gsm8k

Quantisation tips:
  8B  model: 4-bit NF4 uses ~5 GB VRAM  → default --quant 4bit
  70B model: 4-bit NF4 uses ~38 GB VRAM → needs A100-80G or two A40s
             Use --max_seq_len 256 to keep gradient-pass memory in budget
""",
    )
    p.add_argument("--models",       type=str, default="",
                   help="Comma-separated HuggingFace model IDs")
    p.add_argument("--model_group",  type=str, default="",
                   choices=list(MODEL_GROUPS.keys()) + [""],
                   help="Predefined model group (overridden by --models)")
    p.add_argument("--tasks",        type=str, default=",".join(DEFAULT_TASKS),
                   help=f"Comma-separated tasks (default: {','.join(DEFAULT_TASKS)})")
    p.add_argument("--n_samples",    type=int, default=50)
    p.add_argument("--output_dir",   type=str, default="results")
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quant",        type=str, default="4bit",
                   choices=["4bit", "8bit", "none"],
                   help="BitsAndBytes quantisation level (default: 4bit NF4)")
    p.add_argument("--hf_token",     type=str, default=None,
                   help=("HuggingFace access token for gated models (meta-llama/*). "
                         "Also reads HF_TOKEN env var. "
                         "Not needed for NousResearch/* mirrors."))
    p.add_argument("--max_seq_len",  type=int, default=512,
                   help=("Truncate prompts to this many tokens (default: 512). "
                         "Use 256 for 70B models to keep gradient-pass memory "
                         "within a single 80GB GPU."))
    p.add_argument("--mass_coverage", type=float, default=0.90,
                   help=("Per-sample active subgraph: min edges covering this "
                         "fraction of attribution mass (default: 0.90)"))
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # resolve model list
    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model_group:
        model_names = MODEL_GROUPS[args.model_group]
    else:
        model_names = MODEL_GROUPS["llama"]   # new default: Llama-3 family

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Authenticate once for the whole session (reads HF_TOKEN env var too)
    _hf_login(args.hf_token)

    log.info("Device       : %s", args.device)
    log.info("Quant        : %s", args.quant)
    log.info("max_seq_len  : %d", args.max_seq_len)
    log.info("Models (%d)  : %s", len(model_names), model_names)
    log.info("Tasks  (%d)  : %s", len(task_names),  task_names)
    log.info("Samples/task : %d", args.n_samples)
    log.info("mass_coverage: %.0f%%", args.mass_coverage * 100)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log.info("GPU          : %s  (%.1f GB)", p.name, p.total_memory / 1e9)

    results = run_experiment(
        model_names=model_names,
        task_names=task_names,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        device=args.device,
        mass_coverage=args.mass_coverage,
        quant=args.quant,
        max_seq_len=args.max_seq_len,
        hf_token=args.hf_token,
        seed=args.seed,
    )

    csv_path     = os.path.join(args.output_dir, "path_metrics.csv")
    plot_path    = os.path.join(args.output_dir, "entropy_vs_complexity.png")
    scaling_path = os.path.join(args.output_dir, "scaling_curves.png")

    save_csv(results, csv_path)
    plot_results(results, plot_path)
    plot_scaling(results, scaling_path)
    print_summary(results)


if __name__ == "__main__":
    main()
