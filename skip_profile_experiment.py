#!/usr/bin/env python3
"""
skip_profile_experiment.py  —  Skip Profile Analysis: Layer-Level Capability Localization
===========================================================================================

Measures WHERE along the layer stack each LLM capability class concentrates its
compute edges (vs. skip/residual-passthrough edges), using per-layer Attribution
Patching scores as a continuous proxy for "how much work is done at layer ℓ."

Key quantity — layer compute weight for token t at layer ℓ:

    w_t(ℓ) = (attn_score_t[ℓ] + mlp_score_t[ℓ])
             ─────────────────────────────────────
             Σ_ℓ (attn_score_t[ℓ] + mlp_score_t[ℓ])

    High w_t(ℓ)  →  compute-edge dominant at ℓ  (low skip probability)
    Low  w_t(ℓ)  →  skip-edge dominant at ℓ     (high skip probability)

Derived statistics (reported per condition, mean ± std across examples):

    CoM_t = Σ_ℓ ℓ · w_t(ℓ)                       centroid of compute layers
    ES_t  = mean w_t(ℓ)  for ℓ ∈ [0,   L/3)       early-third compute density
    MS_t  = mean w_t(ℓ)  for ℓ ∈ [L/3, 2L/3)      middle-third compute density
    LS_t  = mean w_t(ℓ)  for ℓ ∈ [2L/3, L)        late-third compute density

Four controlled test conditions (built-in datasets, no downloads required):

    Test 1 — Subject-Verb Agreement   (syntax)      → H1: low CoM, high ES
    Test 2 — Factual Recall           (world-knowledge retrieval) → H1: high CoM, high LS
    Test 3a — 1-hop Bridging          (direct recall)             → H2: intermediate CoM
    Test 3b — 2-hop Bridging          (composition)               → H2: mid CoM, high MS
    Test 4a — Induction Copy          (early-layer pattern)       → H3: low CoM
    Test 4b — Compositional Copy      (entity-attribute retrieval) → H3: high CoM

Three falsifiable hypotheses (H1–H3):

    H1  Syntax tokens compute early; retrieval tokens compute late.
        Prediction: CoM(SV) < CoM(2-hop) < CoM(Factual).

    H2  Multi-hop / compositional tokens concentrate compute in the middle third.
        Prediction: MS(2-hop) > MS(1-hop) and MS(2-hop) > MS(Factual).

    H3  Causal ablation of layer regions produces a double dissociation:
        - Zeroing early layers collapses SV-agreement and induction accuracy.
        - Zeroing mid layers collapses 2-hop accuracy (1-hop remains intact).
        - Zeroing late layers collapses factual-recall accuracy.

Outputs (in --output_dir, default: results/skip_profile/):

    skip_profile_heatmap.png    6-row × L-col heatmap of mean w̄_t(ℓ)
    skip_profile_com.png        Center-of-mass bar chart with ±1 SD error bars
    ablation_results.png        Grouped bar chart: accuracy drop per condition × region
    skip_profile_summary.csv    Per-condition CoM, ES, MS, LS statistics
    ablation_results.csv        Accuracy before / after ablation for each condition × region

Usage
-----
    # Quick smoke test (GPT-2, CPU, 5 examples per condition)
    python skip_profile_experiment.py --model gpt2 --device cpu --n_samples 5

    # Full run (Llama-3-8B, 4-bit, GPU)
    python skip_profile_experiment.py \\
        --model NousResearch/Meta-Llama-3-8B --quant 4bit --device cuda

    # GPT-J parallel architecture comparison
    python skip_profile_experiment.py \\
        --model EleutherAI/gpt-j-6b --quant 8bit --device cuda --n_samples 20

    # Gated meta-llama (requires HF token + Meta approval)
    python skip_profile_experiment.py \\
        --model meta-llama/Meta-Llama-3-8B --hf_token YOUR_TOKEN --quant 4bit

    # Skip ablation (profile only, no zeroing experiments — faster)
    python skip_profile_experiment.py --model gpt2 --device cpu --no_ablation
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from transformer_lens import HookedTransformer
except ImportError:
    raise SystemExit(
        "transformer_lens not installed.  Run:  pip install transformer-lens"
    )

from path_analyzer import PathAnalyzer, select_active_edges_by_mass_coverage


# ─────────────────────────────────────────────────────────────────────────────
# Built-in controlled datasets
# Each entry is a dict with keys:  prefix, correct, distractor (optional)
# All correct/distractor tokens are single-word strings starting with a space.
# ─────────────────────────────────────────────────────────────────────────────

# Test 1: Subject-Verb Agreement (SV)
# Attractor noun-phrase contains the *wrong* number, creating a local cue conflict.
# The correct verb must agree with the *subject*, not the attractor.
DATASET_SV = [
    {"prefix": "The key to the cabinet",          "correct": " is",    "distractor": " are"},
    {"prefix": "The keys to the cabinet",          "correct": " are",   "distractor": " is"},
    {"prefix": "The student in the classes",       "correct": " was",   "distractor": " were"},
    {"prefix": "The students in the class",        "correct": " were",  "distractor": " was"},
    {"prefix": "The dog near the trees",           "correct": " runs",  "distractor": " run"},
    {"prefix": "The dogs near the tree",           "correct": " run",   "distractor": " runs"},
    {"prefix": "The book on the shelves",          "correct": " belongs","distractor": " belong"},
    {"prefix": "The books on the shelf",           "correct": " belong","distractor": " belongs"},
    {"prefix": "The report about the accidents",   "correct": " was",   "distractor": " were"},
    {"prefix": "The reports about the accident",   "correct": " were",  "distractor": " was"},
    {"prefix": "The player on the teams",          "correct": " wins",  "distractor": " win"},
    {"prefix": "The players on the team",          "correct": " win",   "distractor": " wins"},
    {"prefix": "The cat behind the boxes",         "correct": " hides", "distractor": " hide"},
    {"prefix": "The cats behind the box",          "correct": " hide",  "distractor": " hides"},
    {"prefix": "The manager of the companies",     "correct": " decides","distractor": " decide"},
    {"prefix": "The managers of the company",      "correct": " decide","distractor": " decides"},
    {"prefix": "The rule about the exceptions",    "correct": " applies","distractor": " apply"},
    {"prefix": "The rules about the exception",    "correct": " apply", "distractor": " applies"},
    {"prefix": "The woman near the windows",       "correct": " stands","distractor": " stand"},
    {"prefix": "The women near the window",        "correct": " stand", "distractor": " stands"},
]

# Test 2: Factual Recall
# Answers are world-knowledge facts the model should have memorized in pre-training.
DATASET_FACTUAL = [
    {"prefix": "The capital of France is",              "correct": " Paris"},
    {"prefix": "The capital of Germany is",             "correct": " Berlin"},
    {"prefix": "The capital of Japan is",               "correct": " Tokyo"},
    {"prefix": "The capital of Italy is",               "correct": " Rome"},
    {"prefix": "The capital of Spain is",               "correct": " Madrid"},
    {"prefix": "The capital of Canada is",              "correct": " Ottawa"},
    {"prefix": "The capital of Australia is",           "correct": " Canberra"},
    {"prefix": "The capital of China is",               "correct": " Beijing"},
    {"prefix": "The chemical symbol for gold is",       "correct": " Au"},
    {"prefix": "Shakespeare wrote Romeo and",           "correct": " Juliet"},
    {"prefix": "Albert Einstein developed the theory of","correct": " relativity"},
    {"prefix": "Water freezes at zero degrees",         "correct": " Celsius"},
    {"prefix": "The Mona Lisa was painted by",          "correct": " Leonardo"},
    {"prefix": "The Great Wall is located in",          "correct": " China"},
    {"prefix": "Apple was co-founded by Steve",         "correct": " Jobs"},
    {"prefix": "The speed of sound travels through",    "correct": " air"},
    {"prefix": "The first planet from the sun is",      "correct": " Mercury"},
    {"prefix": "DNA stands for deoxyribonucleic",       "correct": " acid"},
    {"prefix": "The Eiffel Tower is located in",        "correct": " Paris"},
    {"prefix": "Marie Curie discovered",                "correct": " radium"},
]

# Test 3: Bridging chains (1-hop vs 2-hop)
# Controlled so that surface vocabulary and answer tokens are identical;
# only the number of reasoning hops differs.
DATASET_BRIDGE_1HOP = [
    {"prefix": "Alice was born in Warsaw. Alice was born in",            "correct": " Warsaw"},
    {"prefix": "Bob lives in Tokyo. Bob lives in",                       "correct": " Tokyo"},
    {"prefix": "Carol works at Microsoft. Carol works at",               "correct": " Microsoft"},
    {"prefix": "Dave studied at Oxford. Dave studied at",                "correct": " Oxford"},
    {"prefix": "Eve moved to Sydney. Eve moved to",                      "correct": " Sydney"},
    {"prefix": "Frank grew up in Berlin. Frank grew up in",              "correct": " Berlin"},
    {"prefix": "Grace attended Harvard. Grace attended",                 "correct": " Harvard"},
    {"prefix": "Henry was born in Dublin. Henry was born in",            "correct": " Dublin"},
    {"prefix": "Iris painted in Florence. Iris painted in",              "correct": " Florence"},
    {"prefix": "Jack sailed from Lisbon. Jack sailed from",              "correct": " Lisbon"},
    {"prefix": "Kim trained in Seoul. Kim trained in",                   "correct": " Seoul"},
    {"prefix": "Liam retired to Vienna. Liam retired to",                "correct": " Vienna"},
    {"prefix": "Mia settled in Zurich. Mia settled in",                  "correct": " Zurich"},
    {"prefix": "Nick flew to Cairo. Nick flew to",                       "correct": " Cairo"},
    {"prefix": "Olivia traveled to Nairobi. Olivia traveled to",         "correct": " Nairobi"},
    {"prefix": "Paul relocated to Singapore. Paul relocated to",         "correct": " Singapore"},
    {"prefix": "Quinn visited Bangkok. Quinn visited",                   "correct": " Bangkok"},
    {"prefix": "Rachel interned in Amsterdam. Rachel interned in",       "correct": " Amsterdam"},
    {"prefix": "Sam drove to Denver. Sam drove to",                      "correct": " Denver"},
    {"prefix": "Tara moved to Phoenix. Tara moved to",                   "correct": " Phoenix"},
]

DATASET_BRIDGE_2HOP = [
    {"prefix": "Bob's mother is Alice. Alice was born in Warsaw. Bob's mother was born in",              "correct": " Warsaw"},
    {"prefix": "Carol's boss is Bob. Bob lives in Tokyo. Carol's boss lives in",                         "correct": " Tokyo"},
    {"prefix": "Dave's colleague is Carol. Carol works at Microsoft. Dave's colleague works at",         "correct": " Microsoft"},
    {"prefix": "Eve's mentor is Dave. Dave studied at Oxford. Eve's mentor studied at",                  "correct": " Oxford"},
    {"prefix": "Frank's sister is Eve. Eve moved to Sydney. Frank's sister moved to",                    "correct": " Sydney"},
    {"prefix": "Grace's friend is Frank. Frank grew up in Berlin. Grace's friend grew up in",            "correct": " Berlin"},
    {"prefix": "Henry's advisor is Grace. Grace attended Harvard. Henry's advisor attended",             "correct": " Harvard"},
    {"prefix": "Iris's father is Henry. Henry was born in Dublin. Iris's father was born in",            "correct": " Dublin"},
    {"prefix": "Jack's partner is Iris. Iris painted in Florence. Jack's partner painted in",            "correct": " Florence"},
    {"prefix": "Kim's teacher is Jack. Jack sailed from Lisbon. Kim's teacher sailed from",              "correct": " Lisbon"},
    {"prefix": "Liam's cousin is Kim. Kim trained in Seoul. Liam's cousin trained in",                   "correct": " Seoul"},
    {"prefix": "Mia's uncle is Liam. Liam retired to Vienna. Mia's uncle retired to",                   "correct": " Vienna"},
    {"prefix": "Nick's aunt is Mia. Mia settled in Zurich. Nick's aunt settled in",                     "correct": " Zurich"},
    {"prefix": "Olivia's brother is Nick. Nick flew to Cairo. Olivia's brother flew to",                "correct": " Cairo"},
    {"prefix": "Paul's neighbor is Olivia. Olivia traveled to Nairobi. Paul's neighbor traveled to",    "correct": " Nairobi"},
    {"prefix": "Quinn's roommate is Paul. Paul relocated to Singapore. Quinn's roommate relocated to",   "correct": " Singapore"},
    {"prefix": "Rachel's coach is Quinn. Quinn visited Bangkok. Rachel's coach visited",                 "correct": " Bangkok"},
    {"prefix": "Sam's intern is Rachel. Rachel interned in Amsterdam. Sam's intern interned in",         "correct": " Amsterdam"},
    {"prefix": "Tara's student is Sam. Sam drove to Denver. Tara's student drove to",                   "correct": " Denver"},
    {"prefix": "Alice's nephew is Tara. Tara moved to Phoenix. Alice's nephew moved to",                "correct": " Phoenix"},
]

# Test 4: Induction Copy (literal n-gram repeat) vs. Compositional Copy (attribute retrieval)
DATASET_INDUCTION = [
    {"prefix": "The fox jumped over the fence. The fox jumped over the",         "correct": " fence"},
    {"prefix": "She opened the red door. She opened the red",                    "correct": " door"},
    {"prefix": "He picked up the blue pen. He picked up the blue",               "correct": " pen"},
    {"prefix": "They saw a white cat. They saw a white",                         "correct": " cat"},
    {"prefix": "I placed it on the wooden table. I placed it on the wooden",     "correct": " table"},
    {"prefix": "The bird flew past the old tower. The bird flew past the old",   "correct": " tower"},
    {"prefix": "She wore a green jacket. She wore a green",                      "correct": " jacket"},
    {"prefix": "The child hid behind the tall tree. The child hid behind the tall", "correct": " tree"},
    {"prefix": "He drove through the dark tunnel. He drove through the dark",    "correct": " tunnel"},
    {"prefix": "They drank from the cold stream. They drank from the cold",      "correct": " stream"},
    {"prefix": "The man sat on the hard bench. The man sat on the hard",         "correct": " bench"},
    {"prefix": "She ran across the wet field. She ran across the wet",           "correct": " field"},
    {"prefix": "He climbed up the steep hill. He climbed up the steep",          "correct": " hill"},
    {"prefix": "They swam in the deep lake. They swam in the deep",              "correct": " lake"},
    {"prefix": "The ship sailed past the rocky coast. The ship sailed past the rocky", "correct": " coast"},
    {"prefix": "She painted the brick wall. She painted the brick",              "correct": " wall"},
    {"prefix": "He crossed the narrow bridge. He crossed the narrow",            "correct": " bridge"},
    {"prefix": "They climbed the iron fence. They climbed the iron",             "correct": " fence"},
    {"prefix": "The cat chased the gray mouse. The cat chased the gray",         "correct": " mouse"},
    {"prefix": "She read the thick book. She read the thick",                    "correct": " book"},
]

DATASET_COMPOSITIONAL = [
    {"prefix": "Emma's favorite color is blue. Emma painted her room",           "correct": " blue"},
    {"prefix": "Tom's pet is a dog. Tom walked his",                             "correct": " dog"},
    {"prefix": "Alice's hobby is painting. Alice spent the afternoon",           "correct": " painting"},
    {"prefix": "Dave's car is red. Dave washed his red",                         "correct": " car"},
    {"prefix": "Carol's language is French. Carol spoke to them in",             "correct": " French"},
    {"prefix": "Eve's sport is tennis. Eve went outside to play",                "correct": " tennis"},
    {"prefix": "Frank's instrument is guitar. Frank picked up his",              "correct": " guitar"},
    {"prefix": "Grace's subject is math. Grace solved a",                        "correct": " math"},
    {"prefix": "Henry's city is Paris. Henry flew back to",                      "correct": " Paris"},
    {"prefix": "Iris's season is winter. Iris went skiing in",                   "correct": " winter"},
    {"prefix": "Jack's drink is coffee. Jack poured himself a cup of",           "correct": " coffee"},
    {"prefix": "Kim's flower is rose. Kim planted a",                            "correct": " rose"},
    {"prefix": "Liam's team is Arsenal. Liam cheered for",                       "correct": " Arsenal"},
    {"prefix": "Mia's fruit is mango. Mia bought a ripe",                        "correct": " mango"},
    {"prefix": "Nick's cuisine is Italian. Nick ordered",                        "correct": " Italian"},
    {"prefix": "Olivia's film is Inception. Olivia rewatched",                   "correct": " Inception"},
    {"prefix": "Paul's bird is parrot. Paul trained his",                        "correct": " parrot"},
    {"prefix": "Quinn's metal is gold. Quinn wore a",                            "correct": " gold"},
    {"prefix": "Rachel's stone is ruby. Rachel found a red",                     "correct": " ruby"},
    {"prefix": "Sam's car brand is Tesla. Sam drove his",                        "correct": " Tesla"},
]

# Mapping: condition label → dataset
ALL_CONDITIONS: Dict[str, List[Dict]] = {
    "SV-agreement":       DATASET_SV,
    "Factual-recall":     DATASET_FACTUAL,
    "1-hop bridge":       DATASET_BRIDGE_1HOP,
    "2-hop bridge":       DATASET_BRIDGE_2HOP,
    "Induction copy":     DATASET_INDUCTION,
    "Compositional copy": DATASET_COMPOSITIONAL,
}

# Expected qualitative ordering (for annotation on plots)
CONDITION_ORDER = [
    "SV-agreement",
    "Induction copy",
    "1-hop bridge",
    "2-hop bridge",
    "Compositional copy",
    "Factual-recall",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConditionProfile:
    label:        str
    mean_weights: np.ndarray   # [n_layers] — mean compute weight across examples
    std_weights:  np.ndarray   # [n_layers] — std dev across examples
    com:          float        # mean center of mass
    com_std:      float        # std of center of mass
    es:           float        # mean early-third compute weight
    ms:           float        # mean middle-third
    ls:           float        # mean late-third
    accuracy:     float        # baseline accuracy (no ablation)
    n_samples:    int


@dataclass
class AblationResult:
    condition:    str
    region:       str          # "early" | "mid" | "late"
    baseline_acc: float
    ablated_acc:  float
    acc_drop:     float        # baseline_acc - ablated_acc


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace auth + model loading (mirrors experiment_runner.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _hf_login(token: Optional[str] = None) -> None:
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
        log.debug("No HF token — using cached credentials / public access only.")


def _is_llama_family(model_name: str) -> bool:
    return any(k in model_name.lower()
               for k in ("llama", "mistral", "gemma", "qwen", "phi"))


def load_model(
    model_name: str,
    device:     str = "cuda",
    quant:      str = "4bit",
    hf_token:   Optional[str] = None,
):
    is_llama  = _is_llama_family(model_name)
    nat_dtype = torch.bfloat16 if is_llama else torch.float16
    tl_kwargs = dict(fold_ln=False, center_writing_weights=False,
                     center_unembed=False) if is_llama else {}
    token = (hf_token
             or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    tok_kw = {"token": token} if token else {}

    log.info("Loading  %s  [quant=%s]", model_name, quant)

    if quant == "4bit":
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_cfg,
                device_map="auto", torch_dtype=torch.bfloat16, **tok_kw,
            )
            model = HookedTransformer.from_pretrained(
                model_name, hf_model=hf_model, dtype=torch.bfloat16,
                move_to_device=False, **tl_kwargs, **tok_kw,
            )
            model.eval()
            log.info("  ✓ 4-bit NF4 loaded")
            return model
        except Exception as exc:
            log.warning("  4-bit failed (%s) — trying 8-bit …", exc)

    if quant in ("4bit", "8bit"):
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_cfg,
                device_map="auto", torch_dtype=torch.float16, **tok_kw,
            )
            model = HookedTransformer.from_pretrained(
                model_name, hf_model=hf_model, dtype=torch.float16,
                move_to_device=False, **tl_kwargs, **tok_kw,
            )
            model.eval()
            log.info("  ✓ 8-bit loaded")
            return model
        except Exception as exc:
            log.warning("  8-bit failed (%s) — native dtype …", exc)

    model = HookedTransformer.from_pretrained(
        model_name, dtype=nat_dtype, device=device, **tl_kwargs, **tok_kw)
    model.eval()
    log.info("  ✓ %s (native dtype) loaded", model_name)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis: Skip Profile
# ─────────────────────────────────────────────────────────────────────────────

class SkipProfileAnalyzer:
    """
    Wraps PathAnalyzer to compute per-layer compute weights (skip profiles).

    Skip profile for token t, layer ℓ:

        w_t(ℓ) = (attn_score_t[ℓ] + mlp_score_t[ℓ])
                 ─────────────────────────────────────
                 Σ_ℓ (attn_score_t[ℓ] + mlp_score_t[ℓ])

    This normalizes attribution mass across layers so that w_t sums to 1
    and can be interpreted as a probability distribution over layers:
    "which layer did the network decide to compute at for this token?"
    """

    def __init__(self, model, device: str = "cpu"):
        self.model    = model
        self.device   = device
        self.analyzer = PathAnalyzer(model)
        self.n_layers = model.cfg.n_layers
        self.is_parallel = bool(getattr(model.cfg, "parallel_attn_mlp", False))

        # Layer-third boundaries
        self.early_end = self.n_layers // 3
        self.mid_end   = 2 * self.n_layers // 3
        # early: [0, early_end)    mid: [early_end, mid_end)    late: [mid_end, n_layers)

    # ── Per-example skip profile ─────────────────────────────────────────────

    def compute_layer_weights(
        self,
        tokens:     torch.Tensor,   # [1, seq_len]
        target_pos: int = -1,
    ) -> np.ndarray:
        """
        Return per-layer normalized compute weight w_t(ℓ), shape [n_layers].

        Steps:
          1. Compute AtP attribution scores (attn_scores[ℓ], mlp_scores[ℓ]).
          2. Sum scores at each layer: c[ℓ] = attn[ℓ] + mlp[ℓ].
          3. Normalise: w[ℓ] = c[ℓ] / Σ_ℓ c[ℓ].
          4. If total attribution is zero (degenerate), return uniform distribution.
        """
        try:
            attn_sc, mlp_sc = self.analyzer.compute_attribution_scores(
                tokens, target_pos=target_pos
            )
        except Exception as exc:
            warnings.warn(f"Attribution scoring failed: {exc} — using zeros.")
            return np.full(self.n_layers, 1.0 / self.n_layers)

        attn_np = attn_sc.cpu().float().numpy()
        mlp_np  = mlp_sc.cpu().float().numpy()
        per_layer = attn_np + mlp_np          # sum attn + mlp contribution per layer
        total = per_layer.sum()
        if total <= 1e-12:
            return np.full(self.n_layers, 1.0 / self.n_layers)
        return per_layer / total              # w_t(ℓ), sums to 1

    def compute_profile_stats(self, weights: np.ndarray) -> Dict:
        """
        Compute CoM, ES, MS, LS from a single sample's layer weight vector.

        Returns a dict with: com, es, ms, ls
        """
        layers = np.arange(self.n_layers, dtype=np.float64)
        com = float(np.dot(layers, weights))
        es  = float(weights[:self.early_end].mean())
        ms  = float(weights[self.early_end:self.mid_end].mean())
        ls  = float(weights[self.mid_end:].mean())
        return {"com": com, "es": es, "ms": ms, "ls": ls}

    # ── Accuracy helper ──────────────────────────────────────────────────────

    def _correct_token_idx(self, correct_str: str) -> int:
        """
        Return the vocabulary index of the first sub-token of correct_str.
        Handles both single-token and multi-token answers gracefully.
        """
        try:
            ids = self.model.to_tokens(correct_str, prepend_bos=False)
            return int(ids[0, 0].item())
        except Exception:
            # fallback: try without leading space
            ids = self.model.to_tokens(correct_str.strip(), prepend_bos=False)
            return int(ids[0, 0].item())

    @torch.no_grad()
    def _baseline_accuracy(
        self,
        examples:   List[Dict],
        n_samples:  int,
    ) -> float:
        """
        Greedy accuracy on the first n_samples examples.

        For SV examples with a 'distractor' key: correct if P(correct) > P(distractor).
        For all others: correct if argmax logit == first token of correct string.
        """
        correct = total = 0
        for ex in examples[:n_samples]:
            try:
                tokens = self.model.to_tokens(ex["prefix"], prepend_bos=True).to(self.device)
                logits = self.model(tokens, return_type="logits")
                last_logits = logits[0, -1].float()

                if "distractor" in ex:
                    # Compare P(correct) vs P(distractor)
                    c_idx = self._correct_token_idx(ex["correct"])
                    d_idx = self._correct_token_idx(ex["distractor"])
                    is_correct = (last_logits[c_idx] > last_logits[d_idx]).item()
                else:
                    c_idx      = self._correct_token_idx(ex["correct"])
                    pred_idx   = int(last_logits.argmax().item())
                    is_correct = (pred_idx == c_idx)

                correct += int(is_correct)
                total   += 1
            except Exception as exc:
                warnings.warn(f"Accuracy check failed for example: {exc}")
        return correct / total if total > 0 else float("nan")

    # ── Condition profile (averaged over examples) ───────────────────────────

    def run_condition(
        self,
        examples:  List[Dict],
        label:     str,
        n_samples: int,
    ) -> ConditionProfile:
        """
        Run skip-profile analysis for one condition.

        Returns a ConditionProfile with mean/std weights and derived stats.
        """
        log.info("  Condition: %-22s  (%d examples)", label, min(n_samples, len(examples)))
        weights_all: List[np.ndarray] = []
        com_all, es_all, ms_all, ls_all = [], [], [], []

        for i, ex in enumerate(examples[:n_samples]):
            try:
                tokens = self.model.to_tokens(ex["prefix"], prepend_bos=True).to(self.device)
                w      = self.compute_layer_weights(tokens, target_pos=-1)
                st     = self.compute_profile_stats(w)
                weights_all.append(w)
                com_all.append(st["com"])
                es_all.append(st["es"])
                ms_all.append(st["ms"])
                ls_all.append(st["ls"])
            except Exception as exc:
                warnings.warn(f"Skipping example {i} in '{label}': {exc}")

        if not weights_all:
            dummy = np.full(self.n_layers, 1.0 / self.n_layers)
            return ConditionProfile(label=label, mean_weights=dummy, std_weights=dummy,
                                    com=self.n_layers / 2, com_std=0.0,
                                    es=1/3, ms=1/3, ls=1/3, accuracy=float("nan"),
                                    n_samples=0)

        weights_mat   = np.stack(weights_all, axis=0)    # [n_examples, n_layers]
        mean_w        = weights_mat.mean(axis=0)
        std_w         = weights_mat.std(axis=0)
        baseline_acc  = self._baseline_accuracy(examples, n_samples)

        profile = ConditionProfile(
            label        = label,
            mean_weights = mean_w,
            std_weights  = std_w,
            com          = float(np.mean(com_all)),
            com_std      = float(np.std(com_all)),
            es           = float(np.mean(es_all)),
            ms           = float(np.mean(ms_all)),
            ls           = float(np.mean(ls_all)),
            accuracy     = baseline_acc,
            n_samples    = len(weights_all),
        )
        log.info(
            "    CoM=%.2f  ES=%.3f  MS=%.3f  LS=%.3f  acc=%.1f%%",
            profile.com, profile.es, profile.ms, profile.ls, profile.accuracy * 100,
        )
        return profile

    # ── Causal ablation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def ablation_accuracy(
        self,
        examples:      List[Dict],
        ablate_start:  int,
        ablate_end:    int,
        n_samples:     int,
    ) -> float:
        """
        Measure accuracy after zeroing attn_out + mlp_out for layers
        in [ablate_start, ablate_end).

        This tests H3 (causal double dissociation): if a capability lives
        in a specific layer region, zeroing that region should collapse
        accuracy for that condition but leave other conditions intact.
        """
        fwd_hooks = []
        for layer in range(ablate_start, ablate_end):
            def make_zero_hook(l):
                def _zero(act, hook):
                    return torch.zeros_like(act)
                return _zero
            fwd_hooks.append((f"blocks.{layer}.hook_attn_out", make_zero_hook(layer)))
            if not getattr(self.model.cfg, "attn_only", False):
                fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", make_zero_hook(layer)))

        correct = total = 0
        for ex in examples[:n_samples]:
            try:
                tokens = self.model.to_tokens(ex["prefix"], prepend_bos=True).to(self.device)
                logits = self.model.run_with_hooks(
                    tokens, fwd_hooks=fwd_hooks, return_type="logits"
                )
                last_logits = logits[0, -1].float()

                if "distractor" in ex:
                    c_idx = self._correct_token_idx(ex["correct"])
                    d_idx = self._correct_token_idx(ex["distractor"])
                    is_correct = (last_logits[c_idx] > last_logits[d_idx]).item()
                else:
                    c_idx      = self._correct_token_idx(ex["correct"])
                    pred_idx   = int(last_logits.argmax().item())
                    is_correct = (pred_idx == c_idx)

                correct += int(is_correct)
                total   += 1
            except Exception as exc:
                warnings.warn(f"Ablation accuracy check failed: {exc}")

        return correct / total if total > 0 else float("nan")

    def run_ablation(
        self,
        condition_profiles: List[ConditionProfile],
        examples_by_label:  Dict[str, List[Dict]],
        n_samples:          int,
    ) -> List[AblationResult]:
        """
        For each (condition, region) pair, compute accuracy after zeroing
        the layer region and report the accuracy drop.

        Regions:
            early  [0,          early_end)
            mid    [early_end,  mid_end)
            late   [mid_end,    n_layers)
        """
        regions = {
            "early": (0,              self.early_end),
            "mid":   (self.early_end, self.mid_end),
            "late":  (self.mid_end,   self.n_layers),
        }
        results: List[AblationResult] = []
        for profile in condition_profiles:
            label    = profile.label
            examples = examples_by_label[label]
            for region_name, (r_start, r_end) in regions.items():
                log.info("  Ablation  %-22s  region=%-5s  layers=[%d,%d)",
                         label, region_name, r_start, r_end)
                ablated_acc = self.ablation_accuracy(examples, r_start, r_end, n_samples)
                drop = profile.accuracy - ablated_acc
                results.append(AblationResult(
                    condition    = label,
                    region       = region_name,
                    baseline_acc = profile.accuracy,
                    ablated_acc  = ablated_acc,
                    acc_drop     = drop,
                ))
                log.info("    baseline=%.1f%%  ablated=%.1f%%  drop=%.1f%%",
                         profile.accuracy * 100, ablated_acc * 100, drop * 100)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_CONDITION_COLORS = {
    "SV-agreement":       "#2196F3",   # blue
    "Induction copy":     "#03A9F4",   # light blue
    "1-hop bridge":       "#8BC34A",   # light green
    "2-hop bridge":       "#FF9800",   # orange
    "Compositional copy": "#E91E63",   # pink
    "Factual-recall":     "#9C27B0",   # purple
}
_REGION_COLORS = {"early": "#F44336", "mid": "#FF9800", "late": "#9C27B0"}


def plot_skip_heatmap(
    profiles:   List[ConditionProfile],
    n_layers:   int,
    early_end:  int,
    mid_end:    int,
    output_path: str,
) -> None:
    """
    Heatmap of mean compute weights w̄_t(ℓ) across conditions × layers.

    Rows = conditions (ordered by predicted CoM).
    Cols = layers (0 … L−1).
    Colour = normalized compute weight (viridis; bright = high compute).
    """
    ordered = sorted(profiles, key=lambda p: p.com)
    mat     = np.stack([p.mean_weights for p in ordered], axis=0)

    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.4), 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                   vmin=0.0, vmax=mat.max())
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([p.label for p in ordered], fontsize=10)
    ax.set_xlabel("Transformer layer  ℓ", fontsize=11)
    ax.set_title(
        f"Skip Profile Heatmap — mean compute weight  w̄_t(ℓ)\n"
        f"({n_layers} layers; early/mid/late thirds at {early_end}/{mid_end})",
        fontsize=12,
    )

    # Layer-third boundaries
    for x, label in [(early_end - 0.5, "E|M"), (mid_end - 0.5, "M|L")]:
        ax.axvline(x, color="white", linewidth=2.0, linestyle="--", alpha=0.8)
        ax.text(x + 0.2, len(ordered) - 0.5, label,
                color="white", fontsize=8, va="top", fontweight="bold")

    plt.colorbar(im, ax=ax, label="w̄_t(ℓ)", shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap → %s", output_path)


def plot_com_bars(
    profiles:    List[ConditionProfile],
    n_layers:    int,
    early_end:   int,
    mid_end:     int,
    output_path: str,
) -> None:
    """
    Horizontal bar chart of CoM per condition with ±1 SD error bars.
    Vertical bands shade the early / mid / late thirds.
    """
    ordered = sorted(profiles, key=lambda p: p.com)
    labels  = [p.label for p in ordered]
    coms    = [p.com    for p in ordered]
    stds    = [p.com_std for p in ordered]
    colors  = [_CONDITION_COLORS.get(p.label, "#607D8B") for p in ordered]

    fig, ax = plt.subplots(figsize=(7, max(4, len(profiles) * 0.65)))
    ax.barh(range(len(ordered)), coms, xerr=stds, align="center",
            color=colors, edgecolor="white", linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=4, linewidth=1.2))
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Center of Mass  (CoM = Σ_ℓ ℓ · w_t(ℓ))", fontsize=11)
    ax.set_xlim(0, n_layers)
    ax.set_title("Compute Center of Mass by Condition  (±1 SD)", fontsize=12)

    # Shade thirds
    ax.axvspan(0,         early_end, alpha=0.06, color="red",    label="Early third")
    ax.axvspan(early_end, mid_end,   alpha=0.06, color="orange", label="Mid third")
    ax.axvspan(mid_end,   n_layers,  alpha=0.06, color="purple", label="Late third")
    ax.axvline(early_end, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(mid_end,   color="gray", linewidth=0.8, linestyle="--")

    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved CoM bar chart → %s", output_path)


def plot_ablation(
    ablation_results: List[AblationResult],
    output_path:      str,
) -> None:
    """
    Grouped bar chart: accuracy drop (%) after ablating each layer region.

    Each group = one condition.
    Each bar within a group = one region (early / mid / late).
    H3 prediction: each condition's largest drop should be in its predicted region.
    """
    conditions = list(dict.fromkeys(r.condition for r in ablation_results))
    regions    = ["early", "mid", "late"]
    n_cond     = len(conditions)
    n_reg      = len(regions)
    x          = np.arange(n_cond)
    width      = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n_cond * 1.6), 5))
    for j, region in enumerate(regions):
        drops = []
        for cond in conditions:
            r = next((r for r in ablation_results
                      if r.condition == cond and r.region == region), None)
            drops.append(r.acc_drop * 100 if r else 0.0)

        bars = ax.bar(
            x + (j - 1) * width, drops, width,
            label=f"{region} region",
            color=_REGION_COLORS.get(region, "#607D8B"),
            edgecolor="white", linewidth=0.4, alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy drop  (% points)", fontsize=11)
    ax.set_title(
        "Causal Ablation: Accuracy Drop per Condition × Layer Region\n"
        "(H3 prediction: each condition's largest drop in its predicted region)",
        fontsize=11,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ablation plot → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_csv(
    profiles:    List[ConditionProfile],
    n_layers:    int,
    early_end:   int,
    mid_end:     int,
    output_path: str,
) -> None:
    fieldnames = [
        "condition", "n_samples", "n_layers",
        "early_end", "mid_end",
        "com_mean", "com_std",
        "es_mean", "ms_mean", "ls_mean",
        "accuracy",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in sorted(profiles, key=lambda x: x.com):
            writer.writerow({
                "condition": p.label,
                "n_samples": p.n_samples,
                "n_layers":  n_layers,
                "early_end": early_end,
                "mid_end":   mid_end,
                "com_mean":  round(p.com,     4),
                "com_std":   round(p.com_std, 4),
                "es_mean":   round(p.es,  4),
                "ms_mean":   round(p.ms,  4),
                "ls_mean":   round(p.ls,  4),
                "accuracy":  round(p.accuracy, 4) if not np.isnan(p.accuracy) else "nan",
            })
    log.info("Saved summary CSV → %s", output_path)


def save_ablation_csv(
    ablation_results: List[AblationResult],
    output_path:      str,
) -> None:
    fieldnames = ["condition", "region", "baseline_acc", "ablated_acc", "acc_drop_pct"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ablation_results:
            writer.writerow({
                "condition":     r.condition,
                "region":        r.region,
                "baseline_acc":  round(r.baseline_acc, 4),
                "ablated_acc":   round(r.ablated_acc,  4),
                "acc_drop_pct":  round(r.acc_drop * 100, 2),
            })
    log.info("Saved ablation CSV → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Skip Profile Analysis — Layer-Level Capability Localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="gpt2",
                   help="HuggingFace model ID (e.g. gpt2, NousResearch/Meta-Llama-3-8B)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quant",  default="none",
                   choices=["4bit", "8bit", "none"],
                   help="Quantization mode (requires bitsandbytes for 4bit/8bit)")
    p.add_argument("--hf_token", default=None,
                   help="HuggingFace token for gated models (meta-llama/*)")
    p.add_argument("--n_samples", type=int, default=20,
                   help="Examples per condition to use (max 20 available per condition)")
    p.add_argument("--mass_coverage", type=float, default=0.90,
                   help="Nucleus-style coverage for active-edge selection (0–1)")
    p.add_argument("--output_dir", default="results/skip_profile",
                   help="Directory for output files")
    p.add_argument("--no_ablation", action="store_true",
                   help="Skip causal ablation experiments (faster; profile only)")
    p.add_argument("--conditions", default=None,
                   help="Comma-separated subset of conditions to run "
                        "(default: all 6). E.g. 'SV-agreement,Factual-recall'")
    return p


def main() -> None:
    args   = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Auth ────────────────────────────────────────────────────────────────
    _hf_login(args.hf_token)

    # ── Model ───────────────────────────────────────────────────────────────
    quant = args.quant if args.quant != "none" else "none"
    model = load_model(args.model, device=args.device, quant=quant, hf_token=args.hf_token)
    model.to(args.device)

    spa = SkipProfileAnalyzer(model, device=args.device)

    log.info("Architecture: %d layers  (early=[0,%d)  mid=[%d,%d)  late=[%d,%d))",
             spa.n_layers,
             spa.early_end, spa.early_end, spa.mid_end, spa.mid_end, spa.n_layers)

    # ── Select conditions ────────────────────────────────────────────────────
    if args.conditions:
        selected = [c.strip() for c in args.conditions.split(",")]
        conditions_to_run = {k: v for k, v in ALL_CONDITIONS.items() if k in selected}
        if not conditions_to_run:
            raise ValueError(f"No valid conditions in: {args.conditions}. "
                             f"Valid: {list(ALL_CONDITIONS.keys())}")
    else:
        conditions_to_run = ALL_CONDITIONS

    n = min(args.n_samples, 20)
    log.info("Running %d conditions × %d examples each", len(conditions_to_run), n)

    # ── Profile runs ─────────────────────────────────────────────────────────
    profiles: List[ConditionProfile] = []
    log.info("=" * 60)
    log.info("PHASE 1 — Skip Profile Computation")
    log.info("=" * 60)
    for label, examples in conditions_to_run.items():
        profile = spa.run_condition(examples, label=label, n_samples=n)
        profiles.append(profile)

    # ── Visualisation ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2 — Figures")
    log.info("=" * 60)

    plot_skip_heatmap(
        profiles   = profiles,
        n_layers   = spa.n_layers,
        early_end  = spa.early_end,
        mid_end    = spa.mid_end,
        output_path= os.path.join(args.output_dir, "skip_profile_heatmap.png"),
    )
    plot_com_bars(
        profiles   = profiles,
        n_layers   = spa.n_layers,
        early_end  = spa.early_end,
        mid_end    = spa.mid_end,
        output_path= os.path.join(args.output_dir, "skip_profile_com.png"),
    )

    # ── CSV export ────────────────────────────────────────────────────────────
    save_summary_csv(
        profiles   = profiles,
        n_layers   = spa.n_layers,
        early_end  = spa.early_end,
        mid_end    = spa.mid_end,
        output_path= os.path.join(args.output_dir, "skip_profile_summary.csv"),
    )

    # ── Causal ablation ───────────────────────────────────────────────────────
    if not args.no_ablation:
        log.info("=" * 60)
        log.info("PHASE 3 — Causal Ablation (H3)")
        log.info("=" * 60)
        ablation_results = spa.run_ablation(
            condition_profiles = profiles,
            examples_by_label  = conditions_to_run,
            n_samples          = n,
        )
        plot_ablation(
            ablation_results = ablation_results,
            output_path      = os.path.join(args.output_dir, "ablation_results.png"),
        )
        save_ablation_csv(
            ablation_results = ablation_results,
            output_path      = os.path.join(args.output_dir, "ablation_results.csv"),
        )

        # ── Print summary table ──────────────────────────────────────────────
        log.info("\n%s", "─" * 70)
        log.info("%-24s  %-6s  %7s  %7s  %7s", "Condition", "Region",
                 "Baseline", "Ablated", "Drop%")
        log.info("─" * 70)
        for r in ablation_results:
            log.info("%-24s  %-6s  %6.1f%%  %6.1f%%  %+6.1f%%",
                     r.condition, r.region,
                     r.baseline_acc * 100, r.ablated_acc * 100, r.acc_drop * 100)
    else:
        log.info("Skipping ablation (--no_ablation).")

    # ── Print profile summary table ───────────────────────────────────────────
    log.info("\n%s", "─" * 70)
    log.info("%-24s  %5s  %6s  %6s  %6s  %6s  %6s",
             "Condition", "n", "CoM", "±SD", "ES", "MS", "LS")
    log.info("─" * 70)
    for p in sorted(profiles, key=lambda x: x.com):
        log.info("%-24s  %5d  %6.2f  %6.2f  %6.3f  %6.3f  %6.3f",
                 p.label, p.n_samples, p.com, p.com_std, p.es, p.ms, p.ls)

    log.info("\nAll outputs written to: %s", args.output_dir)


if __name__ == "__main__":
    main()
