#!/usr/bin/env python3
"""
active_subgraph_dot.py
======================
Generate a proper computational graph of the active subgraph for a
transformer on one or more task prompts.

Outputs (per task)
------------------
  <stem>_<task_n>.md   — Mermaid.js flowchart (renders in GitHub / VS Code /
                          https://mermaid.live)
  <stem>_<task_n>.dot  — Graphviz DOT source
  <stem>_<task_n>.svg  — Rendered SVG (requires: pip install graphviz)

Layout
------
  Vertical stack, top → bottom:

    [Embedding]
         |
    ╔══ Layer 0 ══════════════════════════════════════╗  (dashed box)
    ║                                                  ║
    ║   [H0●] [H1○] [H2●] ... [Hn●]                   ║
    ║          │ (active heads only)                   ║
    ║         [W_O]  ← concat + project heads          ║
    ║          │                                       ║
    ║         (+) ←─ ─ ─ ─ (residual skip)             ║  resid_mid
    ║          │                                       ║
    ║        [FFN]                                     ║
    ║          │                                       ║
    ║         (+) ←─ ─ ─ ─ (residual skip)             ║  resid_post
    ╚══════════════════════════════════════════════════╝
         |
    ╔══ Layer 1 ══════════════════════════════════════╗
    ...

  Active head / block  : coloured fill + bold border
  Inactive             : grey
  W_O projection node  : dark-red box (GPT-2: concat all heads, project)
  Residual-add node    : small circle labelled "+"
  Residual skip path   : U-shaped arc left of cluster, labelled α

Usage
-----
  # Single task, GPT-2 (CPU)
  python active_subgraph_dot.py \\
      --model gpt2 --device cpu \\
      --task "Alice is the mother of Bob. Bob is the mother of Carol. \
              Who is Alice's grandchild?" \\
      --label "2-hop reasoning" \\
      --out graphs/2hop

  # Multiple tasks (one .md / .svg per task)
  python active_subgraph_dot.py \\
      --model gpt2 --device cpu \\
      --tasks "The cat sat on the mat." \\
              "What is 7 times 8? The answer is" \\
              "Alice is the mother of Bob. Bob is the mother of Carol. \
               Who is Alice's grandchild?" \\
      --labels "Simple" "Arithmetic" "2-hop" \\
      --out graphs/gpt2

  View the .md file at https://mermaid.live  (paste the content)
  or open the .svg directly in any browser.
"""
from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from transformer_lens import HookedTransformer
from path_analyzer import PathAnalyzer, select_active_edges_by_mass_coverage


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Default magnitude-ratio threshold.  A block is *inactive* (skip-dominant)
# when  ||block_output|| / ||stream_input|| < MAG_THRESHOLD.
# Typical GPT-2 family: active layers ≈ 0.10–0.50, inactive < 0.05.
MAG_THRESHOLD: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Linear interpolate between two hex colours; t in [0, 1]."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


_ATTN_LO  = "#fddbc7"   # pale orange
_ATTN_HI  = "#d62728"   # deep red
_MLP_LO   = "#d1e5f0"   # pale blue
_MLP_HI   = "#1f77b4"   # deep blue
_INACTIVE  = "#f4f4f4"
_SIGMA_BG  = "#ffffff"


def _attn_colour(score_norm: float) -> str:
    return _lerp_hex(_ATTN_LO, _ATTN_HI, max(0.0, min(1.0, score_norm)))


def _mlp_colour(score_norm: float) -> str:
    return _lerp_hex(_MLP_LO, _MLP_HI, max(0.0, min(1.0, score_norm)))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

# Families that require fold_ln=False, center_writing_weights=False,
# center_unembed=False (RMSNorm or non-standard LayerNorm variants).
_NO_FOLD_LN_FAMILIES: tuple = (
    "llama", "mistral", "gemma", "falcon",
    "qwen", "yi", "olmo", "phi", "deepseek",
    "vicuna", "alpaca", "orca", "platypus",
    "command", "cohere", "mixtral",
)

def _is_llama_family(name: str) -> bool:
    return any(k in name.lower() for k in _NO_FOLD_LN_FAMILIES)


def _estimate_param_billions(name: str) -> float:
    """
    Rough parameter-count estimate (in billions) from a model name string.
    Used to decide whether bfloat16 fits on the available GPU before
    attempting 4-bit quantization (which has more TransformerLens edge cases).
    """
    import re
    # Match "7b", "8b", "13b", "70b", "0.5b", "1.5b" etc. (case-insensitive)
    m = re.search(r'(\d+(?:\.\d+)?)\s*[bB](?:\b|-)', name)
    if m:
        return float(m.group(1))
    # Explicit well-known names that don't encode size
    lname = name.lower()
    if "phi-3-mini"    in lname: return 3.8
    if "phi-3-small"   in lname: return 7.0
    if "phi-3-medium"  in lname: return 14.0
    if "phi-2"         in lname: return 2.7
    if "gpt-neox-20b"  in lname: return 20.0
    if "gpt-j"         in lname: return 6.0
    if "gpt2-xl"       in lname: return 1.5
    if "gpt2-large"    in lname: return 0.8
    if "gpt2-medium"   in lname: return 0.35
    if "gpt2"          in lname: return 0.12
    return 7.0   # conservative default: assume ~7B if unknown


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight HuggingFace-native hook runner
# ─────────────────────────────────────────────────────────────────────────────

class _HFHookedModel:
    """
    Drop-in replacement for HookedTransformer that wraps a raw HuggingFace
    model with native PyTorch forward hooks.

    WHY THIS EXISTS
    ---------------
    TransformerLens.from_pretrained() creates a full internal copy of every
    weight tensor to build its own HookedTransformer.  For an 8B bfloat16
    model on a 40 GB A100 this means:
        HF copy on GPU  (16 GB)
      + TL copy on GPU  (16 GB)
      + state-dict during einops conversion (16 GB peak)
      = 48 GB  →  OOM / infinite stall

    This class loads the HF model directly and attaches PyTorch hooks to the
    exact modules that correspond to TransformerLens hook names, so
    compute_per_head_scores and compute_magnitude_ratios work unchanged.

    SUPPORTED HOOK NAMES
    --------------------
      blocks.{l}.hook_resid_pre   — residual stream entering layer l
      blocks.{l}.hook_attn_out    — attention output (after o_proj / W_O)
      blocks.{l}.hook_mlp_out     — MLP / FFN output
      blocks.{l}.attn.hook_z      — per-head value outputs before W_O
                                    (shape: [batch, seq, n_heads, d_head])
    Unrecognised names (e.g. hook_resid_mid) are silently skipped.
    """

    def __init__(self, hf_model, tokenizer, model_name: str):
        self._model    = hf_model
        self._tok      = tokenizer

        c = hf_model.config
        n_heads  = getattr(c, "num_attention_heads", getattr(c, "n_head",  12))
        d_model  = getattr(c, "hidden_size",         getattr(c, "n_embd", 768))
        n_layers = getattr(c, "num_hidden_layers",   getattr(c, "n_layer", 12))

        from types import SimpleNamespace
        self.cfg = SimpleNamespace(
            n_layers          = n_layers,
            n_heads           = n_heads,
            d_model           = d_model,
            d_head            = d_model // n_heads,
            model_name        = model_name,
            attn_only         = False,
            parallel_attn_mlp = bool(getattr(c, "parallel_attn_mlp", False)),
        )

    # ── Interface expected by the rest of the pipeline ────────────────────
    def parameters(self):
        return self._model.parameters()

    def eval(self):
        self._model.eval()
        return self

    def to(self, *a, **kw):
        return self   # already placed by load_model; ignore extra moves

    def to_tokens(self, text: str) -> torch.Tensor:
        return self._tok(text, return_tensors="pt")["input_ids"]

    def __call__(self, tokens, return_type=None):
        """Allow model(tokens, return_type=...) — delegates to run_with_hooks."""
        return self.run_with_hooks(tokens, fwd_hooks=None, return_type=return_type)

    # ── Module accessors ──────────────────────────────────────────────────
    def _layers(self):
        m = self._model
        if hasattr(m, "model")     and hasattr(m.model,     "layers"): return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):   return m.transformer.h
        if hasattr(m, "gpt_neox")  and hasattr(m.gpt_neox,  "layers"): return m.gpt_neox.layers
        raise ValueError(f"Cannot find layer list in {type(m).__name__}")

    @staticmethod
    def _get_attn(layer):
        for name in ("self_attn", "attention", "attn"):
            if hasattr(layer, name): return getattr(layer, name)
        raise ValueError(f"No attention module in {type(layer).__name__}")

    @staticmethod
    def _get_mlp(layer):
        for name in ("mlp", "feed_forward", "ff"):
            if hasattr(layer, name): return getattr(layer, name)
        raise ValueError(f"No MLP module in {type(layer).__name__}")

    # ── Hook runner ───────────────────────────────────────────────────────
    def run_with_hooks(
        self,
        tokens:      torch.Tensor,
        fwd_hooks:   Optional[list] = None,
        return_type: Optional[str]  = None,
    ):
        """
        Run the model with TransformerLens-style hooks.

        fwd_hooks  list of (hook_name, hook_fn)
            hook_fn(activation, hook_obj=None) — return value is ignored;
            hooks are read-only (capture only, no activation patching).
        return_type  "logits" | None
        """
        device = next(self._model.parameters()).device
        tokens = tokens.to(device)
        handles: list = []

        if fwd_hooks:
            layers  = self._layers()
            n_heads = self.cfg.n_heads
            d_head  = self.cfg.d_head

            # Build index: (layer_idx, suffix) → hook_fn
            idx: dict = {}
            for hname, hfn in fwd_hooks:
                parts = hname.split(".")
                if parts[0] != "blocks" or len(parts) < 3:
                    continue
                try:
                    l = int(parts[1])
                except ValueError:
                    continue
                idx[(l, ".".join(parts[2:]))] = hfn

            for (l, suffix), fn in idx.items():
                if l >= len(layers):
                    continue
                layer  = layers[l]
                attn_m = self._get_attn(layer)
                mlp_m  = self._get_mlp(layer)

                if suffix == "hook_resid_pre":
                    def _pre(mod, args, _fn=fn):
                        x = args[0] if isinstance(args, tuple) else args
                        _fn(x, None)
                    handles.append(layer.register_forward_pre_hook(_pre))

                elif suffix == "hook_attn_out":
                    def _ao(mod, args, out, _fn=fn):
                        x = out[0] if isinstance(out, tuple) else out
                        _fn(x, None)
                    handles.append(attn_m.register_forward_hook(_ao))

                elif suffix == "hook_mlp_out":
                    def _mo(mod, args, out, _fn=fn):
                        x = out[0] if isinstance(out, tuple) else out
                        _fn(x, None)
                    handles.append(mlp_m.register_forward_hook(_mo))

                elif suffix in ("attn.hook_z", "hook_z"):
                    # Per-head outputs = input to o_proj before W_O is applied.
                    # Shape: [batch, seq, n_heads * d_head] → view to [b, s, H, Dh]
                    o_proj = getattr(attn_m, "o_proj", None)
                    if o_proj is not None:
                        def _z(mod, args, _fn=fn, _nh=n_heads, _dh=d_head):
                            x = args[0] if isinstance(args, tuple) else args
                            b, s, _ = x.shape
                            _fn(x.view(b, s, _nh, _dh), None)
                        handles.append(o_proj.register_forward_pre_hook(_z))
                # else: silently skip (hook_resid_mid etc. — not used in analysis)

        try:
            out = self._model(tokens)
        finally:
            for h in handles:
                h.remove()

        if return_type == "logits":
            return out.logits if hasattr(out, "logits") else out[0]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Recommended open-source models by parameter scale
# (GPT-4o is closed-source; these are the nearest open alternatives for
#  active-subgraph analysis via TransformerLens)
# ─────────────────────────────────────────────────────────────────────────────
LARGE_MODEL_PRESETS: dict = {
    # key: (hf_repo, n_layers, n_heads, vram_4bit_GB, notes)
    "gpt2-xl":          ("gpt2-xl",                             48, 25,  7,  "No token needed"),
    "gpt-j-6b":         ("EleutherAI/gpt-j-6b",                 28, 16,  4,  "No token needed"),
    "gpt-neox-20b":     ("EleutherAI/gpt-neox-20b",             44, 64, 12,  "No token needed; parallel arch"),
    "pythia-12b":       ("EleutherAI/pythia-12b",               36, 40,  8,  "No token needed; parallel arch"),
    "mistral-7b":       ("mistralai/Mistral-7B-v0.1",           32, 32,  5,  "No token needed"),
    "mixtral-8x7b":     ("mistralai/Mixtral-8x7B-v0.1",         32, 32, 25,  "MoE; no token needed"),
    "gemma-7b":         ("google/gemma-7b",                     28, 16,  5,  "Requires HF token"),
    "llama-3-8b":       ("NousResearch/Meta-Llama-3-8B",        32, 32,  5,  "No token needed (NousResearch)"),
    "llama-3-70b":      ("meta-llama/Meta-Llama-3-70B",         80, 64, 40,  "Requires HF approval; A100 80GB"),
    "llama-3-8b-inst":  ("NousResearch/Meta-Llama-3-8B-Instruct", 32, 32, 5, "Instruction-tuned"),
    "qwen-7b":          ("Qwen/Qwen1.5-7B",                     32, 32,  5,  "No token needed"),
    "phi-3-mini":       ("microsoft/Phi-3-mini-4k-instruct",    32, 32,  3,  "3.8B; no token needed"),
}



def load_model(model_name: str, device: str = "cuda",
               hf_token: Optional[str] = None):
    """
    Load a model for active-subgraph analysis.

    Returns either a HookedTransformer (preferred) or a _HFHookedModel
    (fallback).  Both expose the same interface used by compute_per_head_scores,
    compute_magnitude_ratios, and process_task.

    Loading strategy (tried in order):
    ─────────────────────────────────
    A. HF model loaded to **CPU RAM** in bfloat16, then TransformerLens creates
       its own GPU-resident model by pulling weights from CPU one tensor at a
       time.  Peak GPU VRAM = one model copy (~16 GB for 8B).
       (Previous approach loaded HF to GPU first → 3× model VRAM peak → OOM.)

    B. _HFHookedModel: skip TransformerLens entirely, attach native PyTorch
       hooks directly to the HuggingFace model.  Zero extra VRAM for wrapping.
       Used when (A) fails OR when bfloat16 does not fit (very large models).
       For models > 0.75 × VRAM: HF loads in 4-bit NF4 for strategy B.

    C. CPU fallback for small models (GPT-2 etc.) when no CUDA is available.
    """
    resolved = (hf_token
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if resolved:
        import huggingface_hub
        huggingface_hub.login(token=resolved, add_to_git_credential=False)

    device = device if torch.cuda.is_available() else "cpu"
    extra: dict = {}
    if _is_llama_family(model_name):
        extra = dict(fold_ln=False, center_writing_weights=False,
                     center_unembed=False)

    if device == "cuda":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        param_b = _estimate_param_billions(model_name)
        bf16_gb = param_b * 2.0   # bfloat16 = 2 bytes / param

        print(f"  GPU: {vram_gb:.0f} GB VRAM  |  "
              f"model ~{param_b:.0f}B params ({bf16_gb:.0f} GB bfloat16)")

        # ── Strategy A: bfloat16 via TransformerLens (all-GPU) ───────────────
        # During from_pretrained TL holds three copies simultaneously:
        #   HF model + TL model + state-dict tensors ≈ 3 × bf16_gb peak.
        # Only use this path when the triple-copy peak fits in 90% of VRAM.
        if bf16_gb * 3 < vram_gb * 0.90:
            print(f"  [A] Loading HF model to GPU (bfloat16) …")
            hf_model_a = None
            try:
                hf_model_a = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                tokenizer_a = AutoTokenizer.from_pretrained(model_name)
                hf_model_a.eval()

                print(f"  [A] Wrapping with TransformerLens …")
                tl_model = HookedTransformer.from_pretrained(
                    model_name,
                    hf_model=hf_model_a,
                    tokenizer=tokenizer_a,
                    dtype=torch.bfloat16,
                    move_to_device=True,
                    **extra,
                )
                del hf_model_a
                torch.cuda.empty_cache()
                print(f"  [A] Loaded {model_name} in bfloat16 via TransformerLens.")
                return tl_model

            except Exception as e:
                print(f"  [A] TransformerLens wrapping failed "
                      f"({type(e).__name__}: {e})")
                print(f"  [A] → falling back to native HF hooks (Strategy B).")
                if hf_model_a is not None:
                    del hf_model_a
                torch.cuda.empty_cache()

        # ── Strategy B: native HuggingFace hooks (_HFHookedModel) ─────────────
        # Skips TransformerLens wrapping entirely — zero extra VRAM overhead.
        # Uses PyTorch register_forward_hook / register_forward_pre_hook to
        # capture activations at the same points as TransformerLens hooks.
        print(f"  [B] Loading HF model with native hooks (no TL wrapping) …")
        try:
            if bf16_gb < vram_gb * 0.88:
                # bfloat16 fits — plain loading, best quality
                hf_model_b = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            else:
                # Model is too large for bfloat16 on GPU alone.
                # Strategy: 4-bit NF4 if bitsandbytes >= 0.43, otherwise
                # fall back to bfloat16 + CPU offloading.
                #
                # Why the BnB version gate?
                #   Old BnB (< 0.43) with CPU-offloaded 4-bit layers hits two
                #   cascading bugs: (1) Params4bit.__new__ rejects the new
                #   _is_hf_initialized kwarg from transformers → TypeError;
                #   (2) after patching that away, old BnB re-initialises
                #   every Params4bit on the CPU path and calls .item() on
                #   meta tensors (lazy CPU placeholders) → RuntimeError.
                #   BnB 0.43 added _is_hf_initialized precisely to skip that
                #   re-initialisation.  The meta-tensor error cannot be
                #   monkey-patched away without rewriting BnB internals.
                #   Solution: detect the version and go straight to the
                #   reliable bfloat16+CPU path when BnB is too old.
                #
                # To enable 4-bit (much faster inference):
                #   1. Add  !pip install -qU bitsandbytes  to your notebook
                #   2. Runtime → Restart runtime
                #   3. Re-run
                from transformers import BitsAndBytesConfig
                import gc
                os.environ.setdefault("PYTORCH_ALLOC_CONF",
                                      "expandable_segments:True")

                vram_gib = (torch.cuda.get_device_properties(0).total_memory
                            / (1024 ** 3))
                free_gib = ((torch.cuda.get_device_properties(0).total_memory
                             - torch.cuda.memory_allocated(0))
                            / (1024 ** 3))
                nf4_gib = param_b * 0.5 / 1.073741824  # 4-bit size in GiB

                print(f"  [B] GPU {free_gib:.0f}/{vram_gib:.0f} GiB free …")
                if free_gib < nf4_gib * 1.1:
                    print(f"  [B] WARNING: only {free_gib:.1f} GiB free — "
                          f"restart runtime for a clean GPU.")

                # Check whether this BnB version supports 4-bit + CPU offload.
                # Requires BnB >= 0.43 (added _is_hf_initialized / stable CPU offload).
                # Use version-string comparison — signature introspection is unreliable
                # across BnB refactors (e.g. 0.49.x changed internal class layout).
                try:
                    import bitsandbytes as _bnb_mod
                    _bnb_ver = _bnb_mod.__version__
                    _bnb_ok = (
                        tuple(int(x) for x in _bnb_ver.split(".")[:2]) >= (0, 43)
                    )
                except Exception:
                    _bnb_ok, _bnb_ver = False, "unknown"

                if _bnb_ok:
                    # ── New BnB (≥ 0.43): 4-bit NF4 + CPU overflow ────────
                    print(f"  [B] bitsandbytes {_bnb_ver} — 4-bit NF4 …")
                    bnb = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )

                    def _load_4bit(gq):
                        return AutoModelForCausalLM.from_pretrained(
                            model_name,
                            quantization_config=bnb,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            max_memory={0: f"{gq}GiB", "cpu": "180GiB"},
                        )

                    # Quota must be below nf4_gib (forces CPU layers → BnB
                    # quantization path) and below free_gib/4.2 (bfloat16
                    # fallback = 4× quota must still fit on GPU).
                    gpu_q = min(
                        max(8, int(nf4_gib * 0.95)),
                        max(8, int(free_gib * 0.80)),
                        max(8, int(free_gib / 4.2)),
                    )
                    print(f"  [B] GPU quota {gpu_q} GiB …")
                    try:
                        hf_model_b = _load_4bit(gpu_q)
                    except torch.cuda.OutOfMemoryError:
                        gc.collect(); torch.cuda.empty_cache(); gc.collect()
                        gpu_q2 = max(4, gpu_q // 2)
                        print(f"  [B] OOM → retrying at {gpu_q2} GiB …")
                        hf_model_b = _load_4bit(gpu_q2)

                else:
                    # ── Old BnB (< 0.43): bfloat16 + CPU offloading ───────
                    # No 4-bit attempt — a failed 4-bit load leaves ~70 GiB
                    # of ghost memory that poisons subsequent attempts.
                    print(f"  [B] bitsandbytes {_bnb_ver} (need ≥0.43). "
                          f"4-bit disabled; using bfloat16 + CPU offload.")
                    print(f"  [B] To enable 4-bit: "
                          f"!pip install -qU bitsandbytes  then restart.")

                    # Give GPU 65 % of free VRAM for bfloat16 layers; the
                    # rest spills to CPU RAM (~100 GiB limit keeps us safe
                    # on Colab's 167 GB instances).
                    gpu_bf16 = max(8, int(free_gib * 0.65))
                    print(f"  [B] bfloat16; GPU quota {gpu_bf16} GiB + CPU …")
                    hf_model_b = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        max_memory={0: f"{gpu_bf16}GiB", "cpu": "100GiB"},
                    )

            tokenizer_b = AutoTokenizer.from_pretrained(model_name)
            hf_model_b.eval()
            wrapped = _HFHookedModel(hf_model_b, tokenizer_b, model_name)
            print(f"  [B] Loaded {model_name} via native HF hooks.")
            return wrapped

        except Exception as e:
            print(f"  [B] Native HF loading failed ({type(e).__name__}: {e})")
            torch.cuda.empty_cache()

    # ── Strategy C: CPU via TransformerLens (small models / no GPU) ───────────
    # Only attempt if the model is small enough to load on CPU (< 30 GB
    # bfloat16). For large models all GPU strategies have already been tried.
    if device == "cuda":
        param_b_c = _estimate_param_billions(model_name)
        bf16_gb_c = param_b_c * 2.0
        if bf16_gb_c > 30:
            raise RuntimeError(
                f"{model_name} ({bf16_gb_c:.0f} GB bfloat16) could not be "
                f"loaded via any GPU strategy and is too large for CPU. "
                f"Check that bitsandbytes >= 0.43 is installed "
                f"(run: !pip install -qU bitsandbytes) and restart the "
                f"Colab runtime."
            )
    model = HookedTransformer.from_pretrained(model_name, **extra)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-head attribution scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_head_scores(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_pos: int = -1,
    target_token_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-head attribution scores via gradient × activation on hook_z.

    hook_z has shape [batch, seq, n_heads, d_head].
    Score for head h in layer l:
        s(l, h) = mean_{seq, d_head} | grad_z[l,h] * z[l,h] |

    Falls back to layer-level scoring if hook_z is unavailable.

    Returns
    -------
    head_scores : float ndarray [n_layers, n_heads]
    mlp_scores  : float ndarray [n_layers]
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    is_ao    = bool(getattr(model.cfg, "attn_only", False))

    with torch.no_grad():
        logits_det = model(tokens, return_type="logits")
    if target_token_idx is None:
        target_token_idx = int(logits_det[0, target_pos].argmax())

    # ── KEY FIX: freeze model parameters before backward ──────────────────────
    # Without this, .backward() computes gradients for ALL model weights
    # (345M+ params on CPU for gpt2-medium), making each task take many minutes.
    # We only need gradients w.r.t. the anchor activation, not the weights.
    for p in model.parameters():
        p.requires_grad_(False)

    head_act_store: dict = {}
    mlp_act_store:  dict = {}
    fwd_hooks = []

    def _anchor(act, hook):
        return act.detach().float().requires_grad_(True)
    fwd_hooks.append(("blocks.0.hook_resid_pre", _anchor))

    for l in range(n_layers):
        def _z(act, hook, ll=l):
            act.retain_grad()
            head_act_store[ll] = act
            return act
        fwd_hooks.append((f"blocks.{l}.attn.hook_z", _z))

        if not is_ao:
            def _mlp(act, hook, ll=l):
                act.retain_grad()
                mlp_act_store[ll] = act
                return act
            fwd_hooks.append((f"blocks.{l}.hook_mlp_out", _mlp))

    try:
        with torch.enable_grad():
            logits = model.run_with_hooks(
                tokens, fwd_hooks=fwd_hooks, return_type="logits")
            logits[0, target_pos, target_token_idx].backward()
    except RuntimeError as exc:
        print(f"  Gradient failed: {exc}. Returning zero scores.")
        return np.zeros((n_layers, n_heads)), np.zeros(n_layers)

    head_scores = np.zeros((n_layers, n_heads), dtype=np.float32)
    mlp_scores  = np.zeros(n_layers, dtype=np.float32)

    for l in range(n_layers):
        act  = head_act_store.get(l)
        if act is not None and act.grad is not None:
            a = act.detach().float()       # [1, seq, n_heads, d_head]
            g = act.grad.detach().float()
            # [n_heads] — mean over batch=0, seq, d_head
            per_h = (a * g).abs().mean(dim=[0, 1, 3])
            head_scores[l] = per_h.cpu().numpy()

        ma = mlp_act_store.get(l)
        if ma is not None and ma.grad is not None:
            a = ma.detach().float()
            g = ma.grad.detach().float()
            mlp_scores[l] = float((a * g).abs().mean())

    return head_scores, mlp_scores


def _active_heads(head_scores: np.ndarray, layer: int,
                  rel_threshold: float = 0.15) -> List[bool]:
    """
    Head h in layer l is active if its score ≥ rel_threshold × max score in that layer.
    """
    row = head_scores[layer]
    mx  = row.max()
    if mx <= 0.0:
        return [False] * len(row)
    return (row >= rel_threshold * mx).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Magnitude-ratio scoring  (active / inactive block criterion)
# ─────────────────────────────────────────────────────────────────────────────

def compute_magnitude_ratios(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure per-layer output-to-input magnitude ratio.

    Criterion: a block is *inactive* (skip-dominant) when its transformation
    output is negligible compared to the residual stream it adds into:

    Sequential architecture (GPT-2 style):
        ratio_attn[l] = ||attn_out[l]||₂  /  ||resid_pre[0]||₂   (at prediction pos)
        ratio_mlp[l]  = ||mlp_out[l]||₂   /  ||resid_pre[0]||₂   (at prediction pos)

    Parallel architecture (Pythia / GPT-NeoX style, parallel_attn_mlp=True):
        Same formula; hook_resid_mid does not exist so resid_pre[0] is used
        as reference for both attn and MLP.

    Design rationale (two bugs fixed vs. naive implementation):

    Bug 1 — growing denominator:  Pre-LN models (GPT-2, Pythia) normalise the
        residual stream before each block, so ||attn_out[l]|| is bounded by the
        W_O weight scale regardless of depth.  But ||resid_pre[l]|| grows
        monotonically because every layer adds to the stream.  Dividing a
        bounded numerator by a growing denominator deflates middle-layer ratios
        artificially, producing false "silence zones".
        Fix: use ||resid_pre[0]|| (embedding norm) as a fixed reference for
        all layers.  This is the natural scale of the Pre-LN input budget.

    Bug 2 — position averaging:  Averaging over all sequence positions mixes
        prediction-relevant signal (last token) with unrelated positions.
        Fix: evaluate norms at the last token position only, which is the
        position whose logit we are explaining.

    Returns
    -------
    attn_ratios : float ndarray [n_layers]
    mlp_ratios  : float ndarray [n_layers]  (zeros for attn-only models)
    """
    n_layers    = model.cfg.n_layers
    is_ao       = bool(getattr(model.cfg, "attn_only",        False))
    is_parallel = bool(getattr(model.cfg, "parallel_attn_mlp", False))

    resid_pre_store: dict = {}
    attn_out_store:  dict = {}
    resid_mid_store: dict = {}   # empty for parallel models
    mlp_out_store:   dict = {}

    fwd_hooks = []
    for l in range(n_layers):
        def _rp(act, hook, ll=l):
            resid_pre_store[ll] = act.detach().float()
            return act
        fwd_hooks.append((f"blocks.{l}.hook_resid_pre", _rp))

        def _ao(act, hook, ll=l):
            attn_out_store[ll] = act.detach().float()
            return act
        fwd_hooks.append((f"blocks.{l}.hook_attn_out", _ao))

        if not is_ao:
            if not is_parallel:
                # Sequential: resid_pre → attn → resid_mid → mlp → resid_post
                def _rm(act, hook, ll=l):
                    resid_mid_store[ll] = act.detach().float()
                    return act
                fwd_hooks.append((f"blocks.{l}.hook_resid_mid", _rm))

            def _mo(act, hook, ll=l):
                mlp_out_store[ll] = act.detach().float()
                return act
            fwd_hooks.append((f"blocks.{l}.hook_mlp_out", _mo))

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type=None)

    attn_ratios = np.zeros(n_layers, dtype=np.float32)
    mlp_ratios  = np.zeros(n_layers, dtype=np.float32)

    # ── Reference norm: embedding scale at the prediction position ──────────
    # Using resid_pre[0] at the last token as the fixed reference for ALL
    # layers avoids the growing-denominator artifact of Pre-LN transformers.
    rp0 = resid_pre_store.get(0)
    if rp0 is None:
        return attn_ratios, mlp_ratios                 # hooks failed entirely
    # Last token position = the position whose logit is being explained.
    ref_n = rp0[:, -1, :].norm(dim=-1).mean().item()  # scalar, embedding scale

    for l in range(n_layers):
        ao = attn_out_store.get(l)
        if ao is not None:
            # Norm at the prediction position only (last token).
            # Pre-LN: attn_out magnitude is set by W_O weights, not stream size.
            out_n = ao[:, -1, :].norm(dim=-1).mean().item()
            attn_ratios[l] = out_n / (ref_n + 1e-8)

        if not is_ao:
            mo = mlp_out_store.get(l)
            if mo is not None:
                out_n = mo[:, -1, :].norm(dim=-1).mean().item()
                mlp_ratios[l] = out_n / (ref_n + 1e-8)

    return attn_ratios, mlp_ratios


# ─────────────────────────────────────────────────────────────────────────────
# Mermaid.js generator
# ─────────────────────────────────────────────────────────────────────────────

def _mermaid_node_id(*parts) -> str:
    return "_".join(str(p) for p in parts)


def build_mermaid(
    model_name:   str,
    n_layers:     int,
    n_heads:      int,
    head_scores:  np.ndarray,      # [n_layers, n_heads]
    mlp_scores:   np.ndarray,      # [n_layers]
    active_attn:  List[bool],      # [n_layers] magnitude-based
    active_mlp:   List[bool],      # [n_layers] magnitude-based
    attn_ratios:  np.ndarray,      # [n_layers] ||attn_out||/||resid_pre||
    mlp_ratios:   np.ndarray,      # [n_layers] ||mlp_out||/||resid_mid||
    task_text:    str,
    task_label:   str,
    mass_coverage: float,
    epsilon:      float,
    k_edges:      int,
    is_attn_only: bool = False,
    head_threshold: float = 0.15,
    mag_threshold:  float = MAG_THRESHOLD,
) -> str:
    """Return a Mermaid.js flowchart string."""

    # Normalise scores for colour mapping
    hs_max = head_scores.max() if head_scores.max() > 0 else 1.0
    ms_max = mlp_scores.max()  if mlp_scores.max()  > 0 else 1.0

    lines: List[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    wrapped = textwrap.shorten(task_text, width=70, placeholder=" ...")
    lines += [
        "---",
        f"title: \"{task_label} | {model_name} | coverage={mass_coverage:.0%}  k={k_edges}\"",
        "---",
        "%%{ init: { 'theme': 'base', 'themeVariables': {",
        "    'background':    '#ffffff',",
        "    'primaryColor':  '#f4f4f4',",
        "    'lineColor':     '#555555',",
        "    'fontSize':      '13px'",
        "} } }%%",
        "flowchart TB",
        "",
        "  %% ── Class definitions ─────────────────────────────────────────",
        "  classDef active_head  fill:#d62728,stroke:#9a1a1a,stroke-width:2px,color:#fff",
        "  classDef inactive_head fill:#f4f4f4,stroke:#cccccc,stroke-width:1px,color:#aaa",
        "  classDef active_mlp   fill:#1f77b4,stroke:#1a5276,stroke-width:2px,color:#fff",
        "  classDef inactive_mlp fill:#f4f4f4,stroke:#cccccc,stroke-width:1px,color:#aaa",
        "  classDef sigma        fill:#fff,stroke:#444,stroke-width:1.5px,color:#333,font-size:14px",
        "  classDef wo_node      fill:#a93226,stroke:#7b241c,stroke-width:2px,color:#fff",
        "  classDef io_node      fill:#f0f0f0,stroke:#555,stroke-width:2px,color:#222",
        "",
        "  %% ── Prompt ──────────────────────────────────────────────────────",
        f'  PROMPT[/"📝 {wrapped}"/]:::io_node',
        "  EMBED[\"🔷 Embedding\"]:::io_node",
        "  PROMPT --> EMBED",
        "",
    ]

    prev_sum = "EMBED"   # ID of node feeding into next layer

    # ── Layers ───────────────────────────────────────────────────────────────
    for l in range(n_layers):
        n_act_h  = 0
        act_h    = _active_heads(head_scores, l, head_threshold)

        wo_id    = _mermaid_node_id("WO", l)     # W_O output projection
        sa_id    = _mermaid_node_id("SA", l)     # "+" residual add: resid_mid = resid_pre + attn_out
        sm_id    = _mermaid_node_id("SM", l)     # "+" residual add: resid_post = resid_mid + mlp_out
        ffn_id   = _mermaid_node_id("FFN", l)

        # Layer summary for subgraph label (includes magnitude ratios)
        n_act_h   = sum(act_h) if active_attn[l] else 0
        head_info = f"{n_act_h}/{n_heads} heads" if not is_attn_only else ""
        a_mark = f"✓ r={attn_ratios[l]:.2f}" if active_attn[l] else f"✗ r={attn_ratios[l]:.2f}"
        m_mark = f"✓ r={mlp_ratios[l]:.2f}"  if active_mlp[l]  else f"✗ r={mlp_ratios[l]:.2f}"
        mlp_info  = f"FFN {m_mark}"
        sg_label  = (f"Layer {l}  |  Attn {a_mark}  {mlp_info}"
                     if not is_attn_only else f"Layer {l}  |  Attn {a_mark}")

        lines.append(f"  subgraph L{l}[\"{sg_label}\"]")
        lines.append(f"    direction TB")

        # Per-head nodes
        head_ids = []
        for h in range(n_heads):
            hid    = _mermaid_node_id("H", l, h)
            head_ids.append(hid)
            if active_attn[l] and act_h[h]:
                score_n = float(head_scores[l, h]) / hs_max
                lines.append(f"    {hid}[\"H{h} ●\"]:::active_head")
            else:
                lines.append(f"    {hid}[\"H{h} ○\"]:::inactive_head")

        # W_O output-projection node (only when magnitude-active)
        # Correctly models GPT-2: heads are concat+projected before residual add
        if active_attn[l]:
            lines.append(f"    {wo_id}[\"W_O\"]:::wo_node")

        # "+" residual-add node: resid_mid = resid_pre + attn_out
        lines.append(f"    {sa_id}((\"+\")):::sigma")

        # MLP / FFN
        if not is_attn_only:
            m_norm = float(mlp_scores[l]) / ms_max
            if active_mlp[l]:
                lines.append(f"    {ffn_id}[\"FFN  L{l}\"]:::active_mlp")
            else:
                lines.append(f"    {ffn_id}[\"FFN  L{l}\"]:::inactive_mlp")
            # "+" residual-add node: resid_post = resid_mid + mlp_out
            lines.append(f"    {sm_id}((\"+\")):::sigma")

        lines.append("  end")
        lines.append("")

        # ── Edges into this layer ─────────────────────────────────────────
        # Skip / residual arc:
        #   block INACTIVE (ratio ≤ threshold) → thick solid  (skip is dominant)
        #   block ACTIVE   (ratio >  threshold) → thin dashed  (block transforms)
        attn_skip = (f"  {prev_sum} -- α r={attn_ratios[l]:.2f} --> {sa_id}"
                     if active_attn[l]
                     else f"  {prev_sum} -. α r={attn_ratios[l]:.2f} .-> {sa_id}")
        lines.append(attn_skip)

        for h, hid in enumerate(head_ids):
            if active_attn[l] and act_h[h]:
                lines.append(f"  {prev_sum} --> {hid}")
                lines.append(f"  {hid} --> {wo_id}")  # heads → W_O, not directly to "+"

        # W_O projection output → residual-add node
        if active_attn[l] and sum(act_h) > 0:
            lines.append(f"  {wo_id} --> {sa_id}")

        if not is_attn_only:
            mlp_skip = (f"  {sa_id} -- α r={mlp_ratios[l]:.2f} --> {sm_id}"
                        if active_mlp[l]
                        else f"  {sa_id} -. α r={mlp_ratios[l]:.2f} .-> {sm_id}")
            lines.append(mlp_skip)
            if active_mlp[l]:
                lines.append(f"  {sa_id} --> {ffn_id}")
                lines.append(f"  {ffn_id} --> {sm_id}")
            prev_sum = sm_id
        else:
            prev_sum = sa_id

        lines.append("")

    # ── Output ───────────────────────────────────────────────────────────────
    lines += [
        "  OUT[\"🔶 Logits / Output\"]:::io_node",
        f"  {prev_sum} --> OUT",
        "",
        f"  %% Stats: k={k_edges} active edges  epsilon={epsilon:.5f}  "
        f"coverage={mass_coverage:.0%}",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Graphviz DOT generator
# ─────────────────────────────────────────────────────────────────────────────

def build_dot(
    model_name:   str,
    n_layers:     int,
    n_heads:      int,
    head_scores:  np.ndarray,
    mlp_scores:   np.ndarray,
    active_attn:  List[bool],      # magnitude-based: ratio > mag_threshold
    active_mlp:   List[bool],
    attn_ratios:  np.ndarray,      # [n_layers] ||attn_out||/||resid_pre||
    mlp_ratios:   np.ndarray,      # [n_layers] ||mlp_out||/||resid_mid||
    task_text:    str,
    task_label:   str,
    mass_coverage: float,
    epsilon:      float,
    k_edges:      int,
    is_attn_only: bool = False,
    head_threshold: float = 0.15,
    mag_threshold:  float = MAG_THRESHOLD,
) -> str:
    """Return a Graphviz DOT string."""

    hs_max = head_scores.max() if head_scores.max() > 0 else 1.0
    ms_max = mlp_scores.max()  if mlp_scores.max()  > 0 else 1.0
    wrapped = textwrap.shorten(task_text, width=60, placeholder=" ...")

    # Residual stream colour (teal-green, distinct from red attn & blue MLP)
    _RESID_COL  = "#17a589"   # active residual connection
    _RESID_NODE = "#a9cce3"   # stream checkpoint node fill

    lines: List[str] = []
    lines += [
        'digraph active_subgraph {',
        '  rankdir=TB;',
        '  splines=ortho;',            # orthogonal routing → U-shaped skip arcs via :w ports
        '  nodesep=0.45;',
        '  ranksep=0.60;',
        '  bgcolor="#ffffff";',
        '  dpi=180;',
        f'  label="{task_label} | {model_name} | '
        f'coverage={mass_coverage:.0%}  k={k_edges}\\n\\"{wrapped}\\"";',
        '  labelloc=t; fontsize=11; fontname="Helvetica";',
        '',
        '  // Global node defaults',
        '  node [fontname="Helvetica", fontsize=9];',
        '  edge [fontname="Helvetica", fontsize=8];',
        '',
        '  // ── Legend (bottom-right) ──────────────────────────────────────',
        '  subgraph cluster_legend {',
        '    label="Legend"; style=solid; color="#cccccc"; bgcolor="#fefefe";',
        '    fontsize=8; fontname="Helvetica";',
        '    LEG_A  [label="Attn head (active)", shape=ellipse,'
        '            fillcolor="#d62728", style=filled, fontcolor=white, fontsize=7];',
        '    LEG_AI [label="Attn head (inactive)", shape=ellipse,'
        '            fillcolor="#f4f4f4", style=filled, fontcolor="#aaa", fontsize=7];',
        '    LEG_F  [label="FFN (active)", shape=box,'
        '            fillcolor="#1f77b4", style=filled, fontcolor=white, fontsize=7];',
        '    LEG_FI [label="FFN (inactive)", shape=box,'
        '            fillcolor="#f4f4f4", style=filled, fontcolor="#aaa", fontsize=7];',
        '    LEG_WO [label="W_O projection\\n(concat+project\\nall heads)", shape=box,'
        '            fillcolor="#a93226", style=filled, fontcolor=white, fontsize=7];',
        '    LEG_PA [label="+ node\\n(residual add)", shape=circle,'
        '            fillcolor="#f5f5dc", style=filled, fontcolor="#333", fontsize=7];',
        '    LEG_R  [label="Residual stream\\ncheckpoint", shape=diamond,'
        f'            fillcolor="{_RESID_NODE}", style=filled, fontcolor="#1a5276", fontsize=7];',
        '    LEG_SK [label="Skip arc\\n(block inactive,\\nthick teal)", shape=plaintext,'
        '            fontsize=7, fontcolor="#17a589"];',
        '    LEG_SA [label="Skip arc\\n(block active,\\nthin grey dashed)", shape=plaintext,'
        '            fontsize=7, fontcolor="#999999"];',
        '    { rank=same; LEG_A; LEG_AI; LEG_WO; LEG_PA; LEG_F; LEG_FI; LEG_R; LEG_SK; LEG_SA; }',
        '  }',
        '',
    ]

    def dot_node(nid, label, shape="box", style="filled",
                 fillcolor="#f4f4f4", color="#cccccc",
                 penwidth=1.0, fontcolor="#888888", **kw):
        extras = " ".join(f'{k}="{v}"' for k, v in kw.items())
        return (f'  {nid} [label="{label}", shape={shape}, style="{style}", '
                f'fillcolor="{fillcolor}", color="{color}", '
                f'penwidth={penwidth:.1f}, fontcolor="{fontcolor}" {extras}];')

    def dot_edge(src, dst, style="solid", color="#555555",
                 penwidth=1.5, label="", constraint=True, **kw):
        attrs = [f'style="{style}"', f'color="{color}"',
                 f'penwidth={penwidth:.1f}']
        if label:
            attrs.append(f'label="{label}"')
        if not constraint:
            attrs.append('constraint=false')
        attrs += [f'{k}="{v}"' for k, v in kw.items()]
        return f'  {src} -> {dst} [{", ".join(attrs)}];'

    def residual_stream_node(nid: str, l_label: str) -> str:
        """Diamond node representing a residual stream checkpoint."""
        return dot_node(nid, l_label, shape="diamond",
                        fillcolor=_RESID_NODE, color="#1a5276",
                        penwidth=1.8, fontcolor="#1a5276",
                        width="0.45", height="0.45", fixedsize="true")

    # ── Residual stream checkpoints (OUTSIDE clusters) ────────────────────────
    # One diamond per layer boundary: RS_0 = before L0, RS_l = between L(l-1) and Ll
    # These form the visible green backbone.
    stream_ids = []
    for l in range(n_layers + 1):
        rs = f"RS_{l}"
        stream_ids.append(rs)
        lbl = "in" if l == 0 else ("out" if l == n_layers else f"r{l}")
        lines.append(residual_stream_node(rs, lbl))
    lines.append("")

    # Embed feeds the first stream node
    lines.append(dot_node("EMBED", "Embedding", shape="box",
                           fillcolor="#e8e8e8", color="#555555",
                           penwidth=1.5, fontcolor="#222222"))
    lines.append(dot_edge("EMBED", "RS_0",
                           color=_RESID_COL, penwidth=3.0,
                           arrowhead="normal"))
    lines.append("")

    for l in range(n_layers):
        act_h  = _active_heads(head_scores, l, head_threshold)
        n_ah   = sum(act_h) if active_attn[l] else 0
        wo_id  = f"WO_{l}"    # W_O output projection (concat + project all heads)
        sa_id  = f"SA_{l}"    # residual add: resid_pre + attn_out = resid_mid
        sm_id  = f"SM_{l}"    # residual add: resid_mid + mlp_out = resid_post
        ffn_id = f"FFN_{l}"
        rs_in  = f"RS_{l}"       # stream checkpoint entering this layer
        rs_out = f"RS_{l+1}"     # stream checkpoint leaving this layer

        a_mark = f"\u2713 r={attn_ratios[l]:.2f}" if active_attn[l] else f"\u2717 r={attn_ratios[l]:.2f}"
        m_mark = f"\u2713 r={mlp_ratios[l]:.2f}"  if active_mlp[l]  else f"\u2717 r={mlp_ratios[l]:.2f}"
        sg_label = (f"Layer {l}  |  attn {a_mark}   ffn {m_mark}"
                    if not is_attn_only
                    else f"Layer {l}  |  attn {a_mark}")

        lines.append(f'  subgraph cluster_L{l} {{')
        lines.append(f'    label="{sg_label}"; style=dashed; '
                     f'color="#888888"; bgcolor="#fafafa"; '
                     f'fontsize=9; fontname="Helvetica-Oblique";')

        # ── Head nodes ───────────────────────────────────────────────────
        lines.append(f'    {{ rank=same;')
        for h in range(n_heads):
            hid = f"H_{l}_{h}"
            if active_attn[l] and act_h[h]:
                fc = _attn_colour(float(head_scores[l, h]) / hs_max)
                lines.append("    " + dot_node(
                    hid, f"H{h}", shape="ellipse",
                    fillcolor=fc, color="#9a1a1a",
                    penwidth=2.0, fontcolor="#ffffff"))
            else:
                lines.append("    " + dot_node(
                    hid, f"H{h}", shape="ellipse",
                    fillcolor=_INACTIVE, color="#cccccc",
                    penwidth=0.8, fontcolor="#aaaaaa"))
        lines.append(f'    }}')

        # ── W_O output-projection node (GPT-2 architecture) ──────────────
        # In GPT-2 the per-head outputs z_h are concatenated and linearly
        # projected:  attn_out = Concat(z_0..z_H) W_O
        # This is distinct from the residual addition performed at SA below.
        # W_O is only shown when the attention block is magnitude-active.
        if active_attn[l]:
            lines.append("    " + dot_node(
                wo_id, "W_O", shape="box",
                fillcolor="#a93226", color="#7b241c",
                penwidth=1.8, fontcolor="#ffffff"))

        # ── Residual-add node: resid_mid = resid_pre + attn_out ──────────
        # Labelled "+" (not "\u03a3") to clearly show this is the skip addition.
        lines.append("    " + dot_node(
            sa_id, "+", shape="circle",
            fillcolor=_SIGMA_BG, color="#444444",
            penwidth=1.5, fontcolor="#333333",
            width="0.35", height="0.35", fixedsize="true"))

        if not is_attn_only:
            m_norm = float(mlp_scores[l]) / ms_max
            ffn_label = f"FFN  L{l}\\nr={mlp_ratios[l]:.2f}"
            if active_mlp[l]:
                fc_m = _mlp_colour(m_norm)
                lines.append("    " + dot_node(
                    ffn_id, ffn_label, shape="box",
                    fillcolor=fc_m, color="#1a5276",
                    penwidth=2.0, fontcolor="#ffffff"))
            else:
                lines.append("    " + dot_node(
                    ffn_id, ffn_label, shape="box",
                    fillcolor=_INACTIVE, color="#cccccc",
                    penwidth=0.8, fontcolor="#aaaaaa"))

            # Residual-add node: resid_post = resid_mid + mlp_out
            lines.append("    " + dot_node(
                sm_id, "+", shape="circle",
                fillcolor=_SIGMA_BG, color="#444444",
                penwidth=1.5, fontcolor="#333333",
                width="0.35", height="0.35", fixedsize="true"))

        lines.append("  }")   # end cluster
        lines.append("")

        # ── Compute edges ──────────────────────────────────────────────────
        # GPT-2 attention: resid_pre → each head → W_O → [+] (residual add)
        # Heads flow into W_O (output projection), NOT directly into the "+" node.
        for h in range(n_heads):
            hid = f"H_{l}_{h}"
            if active_attn[l] and act_h[h]:
                lines.append(dot_edge(rs_in, hid,
                                       color="#d62728", penwidth=1.6,
                                       weight="2"))
                lines.append(dot_edge(hid, wo_id,
                                       color="#d62728", penwidth=1.6,
                                       weight="2"))

        # W_O output feeds into the residual-add "+" node
        if active_attn[l] and n_ah > 0:
            lines.append(dot_edge(wo_id, sa_id,
                                   color="#d62728", penwidth=1.8,
                                   weight="2"))

        if not is_attn_only:
            if active_mlp[l]:
                lines.append(dot_edge(sa_id, ffn_id,
                                       color="#1f77b4", penwidth=1.8,
                                       weight="2"))
                lines.append(dot_edge(ffn_id, sm_id,
                                       color="#1f77b4", penwidth=1.8,
                                       weight="2"))

        # ── Residual / skip arcs  ─────────────────────────────────────
        # U-shape routing via splines=ortho + :w (west) ports.
        # Graphviz routes:  RS:w → LEFT → DOWN → RIGHT → SA:w
        # giving a clean U around the LEFT side of each layer cluster.
        #
        # Arc style encodes block activity (magnitude-ratio criterion):
        #   block INACTIVE (ratio ≤ threshold) → thick teal  = skip is dominant
        #   block ACTIVE   (ratio >  threshold) → thin grey dashed = block transforms
        if active_attn[l]:
            lines.append(dot_edge(f"{rs_in}:w", f"{sa_id}:w",
                                   style="dashed", color="#999999",
                                   penwidth=1.2,
                                   label=f"\u03b1 {attn_ratios[l]:.2f}",
                                   constraint=False, weight="0",
                                   arrowhead="open"))
        else:
            lines.append(dot_edge(f"{rs_in}:w", f"{sa_id}:w",
                                   style="bold", color=_RESID_COL,
                                   penwidth=3.5,
                                   label=f"\u03b1 {attn_ratios[l]:.2f}",
                                   constraint=False, weight="0",
                                   arrowhead="open"))

        if not is_attn_only:
            if active_mlp[l]:
                lines.append(dot_edge(f"{sa_id}:w", f"{sm_id}:w",
                                       style="dashed", color="#999999",
                                       penwidth=1.2,
                                       label=f"\u03b1 {mlp_ratios[l]:.2f}",
                                       constraint=False, weight="0",
                                       arrowhead="open"))
            else:
                lines.append(dot_edge(f"{sa_id}:w", f"{sm_id}:w",
                                       style="bold", color=_RESID_COL,
                                       penwidth=3.5,
                                       label=f"\u03b1 {mlp_ratios[l]:.2f}",
                                       constraint=False, weight="0",
                                       arrowhead="open"))
            # Backbone: Σ_mlp ──teal──> RS_out
            lines.append(dot_edge(sm_id, rs_out,
                                   color=_RESID_COL, penwidth=3.0,
                                   weight="10", arrowhead="normal"))
        else:
            lines.append(dot_edge(sa_id, rs_out,
                                   color=_RESID_COL, penwidth=3.0,
                                   weight="10", arrowhead="normal"))

        lines.append("")

    # ── Output ───────────────────────────────────────────────────────────────
    lines.append(dot_node("OUT", "Logits / Output", shape="box",
                           fillcolor="#e8e8e8", color="#555555",
                           penwidth=1.5, fontcolor="#222222"))
    lines.append(dot_edge(f"RS_{n_layers}", "OUT",
                           penwidth=3.0, color=_RESID_COL,
                           arrowhead="normal"))
    lines.append("")
    lines.append(f'  // Stats: k={k_edges}  epsilon={epsilon:.5f}'
                 f'  coverage={mass_coverage:.0%}')
    lines.append("}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Compact DOT generator  (large-model summary mode — no individual head nodes)
# ─────────────────────────────────────────────────────────────────────────────

def build_compact_dot(
    model_name:    str,
    n_layers:      int,
    n_heads:       int,
    head_scores:   np.ndarray,
    mlp_scores:    np.ndarray,
    active_attn:   List[bool],
    active_mlp:    List[bool],
    attn_ratios:   np.ndarray,
    mlp_ratios:    np.ndarray,
    task_text:     str,
    task_label:    str,
    mass_coverage: float,
    epsilon:       float,
    k_edges:       int,
    is_attn_only:  bool = False,
    head_threshold: float = 0.15,
    mag_threshold:  float = MAG_THRESHOLD,
    layer_window:   Optional[Tuple[int, int]] = None,
    **_,
) -> str:
    """
    Compact layer-summary Graphviz DOT graph for large models.

    Draws one [Attn] box + one [FFN] box per layer instead of individual head
    ellipses.  Scales to 80+ layer models (Llama-3-70B, GPT-NeoX-20B, etc.)
    without overwhelming Graphviz.

    Node colour encodes the magnitude ratio (heat-map: pale → deep).
    Skip arc style encodes block activity:
      - thick teal  = skip-dominant / block inactive  (ratio < threshold)
      - thin dashed = block active  (ratio >= threshold)

    layer_window : Optional (first_n, last_n).
        When set, only the first `first_n` and last `last_n` layers are
        rendered, separated by a '…' gap node.  Useful for 80-layer models.
    """
    _RESID_COL = "#17a589"  # teal = skip-dominant path

    ar_max  = max(float(attn_ratios.max()), 1e-6)
    mr_max  = max(float(mlp_ratios.max()),  1e-6) if not is_attn_only else 1.0
    wrapped = textwrap.shorten(task_text, width=70, placeholder=" ...")

    # ── Determine which layers to draw ───────────────────────────────────────
    gap_after: Optional[int] = None   # layer index after which to insert '...'
    if layer_window is not None:
        fn, ln = layer_window
        if fn + ln >= n_layers:
            layers_to_show = list(range(n_layers))
        else:
            layers_to_show = list(range(fn)) + list(range(n_layers - ln, n_layers))
            gap_after = fn - 1
    else:
        layers_to_show = list(range(n_layers))

    n_act_a = sum(active_attn[l] for l in layers_to_show)
    n_act_m = (sum(active_mlp[l] for l in layers_to_show)
               if not is_attn_only else 0)
    omit_note = (
        f"  (showing {len(layers_to_show)}/{n_layers} layers)"
        if len(layers_to_show) < n_layers else ""
    )

    header = (
        f"{task_label} | {model_name} | {n_layers}L × {n_heads}H{omit_note}\\n"
        f'\\"{wrapped}\\"\\n'
        f"Active attn: {n_act_a}/{n_layers}   "
        f"Active FFN: {n_act_m}/{n_layers}   "
        f"threshold: {mag_threshold}"
    )

    lines: List[str] = [
        "digraph compact_subgraph {",
        f'  label="{header}";',
        '  labelloc=t; fontsize=10; fontname="Helvetica";',
        "  rankdir=TB;",
        "  splines=curved;",
        "  nodesep=0.25; ranksep=0.50;",
        '  node [fontname="Helvetica" fontsize=9 style=filled];',
        '  edge [fontname="Helvetica" fontsize=7];',
        "",
        '  EMB [label="Embedding" shape=box fillcolor="#e8f5e9" '
        'color="#388e3c" penwidth=1.5 fontcolor="#1b5e20"];',
        "",
    ]

    gap_inserted = False

    # ── Node declarations ────────────────────────────────────────────────────
    for l in layers_to_show:
        a_id  = f"A{l}"
        sa_id = f"SA{l}"
        f_id  = f"F{l}"
        sm_id = f"SM{l}"

        # Attn block
        if active_attn[l]:
            t      = min(1.0, float(attn_ratios[l]) / ar_max)
            a_fill = _lerp_hex(_ATTN_LO, _ATTN_HI, t)
            a_bord, a_pw, a_fc = "#7b241c", "2.0", "#ffffff"
        else:
            a_fill, a_bord, a_pw, a_fc = _INACTIVE, "#cccccc", "0.8", "#888888"

        # Count active heads in this layer (informational label)
        row_max = float(head_scores[l].max()) if head_scores[l].max() > 0 else 1e-9
        n_ah = sum(1 for h in range(n_heads)
                   if head_scores[l, h] >= head_threshold * row_max)
        a_label = f"Attn L{l}  ({n_ah}/{n_heads} heads)\\nr={attn_ratios[l]:.2f}"
        lines.append(
            f'  {a_id} [label="{a_label}" shape=box '
            f'fillcolor="{a_fill}" color="{a_bord}" penwidth={a_pw} '
            f'fontcolor="{a_fc}"];'
        )
        lines.append(
            f'  {sa_id} [label="+" shape=circle fillcolor="{_SIGMA_BG}" '
            f'color="#444444" penwidth=1.5 width=0.30 height=0.30 '
            f'fixedsize=true fontcolor="#333333"];'
        )

        if not is_attn_only:
            if active_mlp[l]:
                t      = min(1.0, float(mlp_ratios[l]) / mr_max)
                f_fill = _lerp_hex(_MLP_LO, _MLP_HI, t)
                f_bord, f_pw, f_fc = "#1a5276", "2.0", "#ffffff"
            else:
                f_fill, f_bord, f_pw, f_fc = _INACTIVE, "#cccccc", "0.8", "#888888"

            f_label = f"FFN L{l}\\nr={mlp_ratios[l]:.2f}"
            lines.append(
                f'  {f_id} [label="{f_label}" shape=box '
                f'fillcolor="{f_fill}" color="{f_bord}" penwidth={f_pw} '
                f'fontcolor="{f_fc}"];'
            )
            lines.append(
                f'  {sm_id} [label="+" shape=circle fillcolor="{_SIGMA_BG}" '
                f'color="#444444" penwidth=1.5 width=0.30 height=0.30 '
                f'fixedsize=true fontcolor="#333333"];'
            )

        # Gap node (declared once, after the first section ends)
        if gap_after is not None and l == gap_after and not gap_inserted:
            lines.append(
                '  GAP [label="⋮  (middle layers omitted)  ⋮" '
                'shape=plaintext fontsize=12 fontcolor="#999999" style=""];'
            )
            gap_inserted = True

        lines.append("")

    # ── Edge declarations ─────────────────────────────────────────────────────
    lines.append("  // ── Edges ──")
    prev = "EMB"
    gap_edge_done = False

    for l in layers_to_show:
        a_id  = f"A{l}"
        sa_id = f"SA{l}"
        f_id  = f"F{l}"
        sm_id = f"SM{l}"

        # Main chain
        lines.append(f'  {prev} -> {a_id} [color="#555555" penwidth=1.4];')
        lines.append(f'  {a_id} -> {sa_id} [color="#d62728" penwidth=1.5];')

        # Attention skip arc
        if active_attn[l]:
            lines.append(
                f'  {prev} -> {sa_id} [style=dashed color="#aaaaaa" penwidth=1.0 '
                f'label="α {attn_ratios[l]:.2f}" '
                f'constraint=false weight=0 arrowhead=open fontsize=7];'
            )
        else:
            lines.append(
                f'  {prev} -> {sa_id} [style=bold color="{_RESID_COL}" penwidth=3.0 '
                f'label="α {attn_ratios[l]:.2f}" '
                f'constraint=false weight=0 arrowhead=open fontsize=7];'
            )

        if not is_attn_only:
            lines.append(f'  {sa_id} -> {f_id} [color="#555555" penwidth=1.4];')
            lines.append(f'  {f_id} -> {sm_id} [color="#1f77b4" penwidth=1.5];')
            if active_mlp[l]:
                lines.append(
                    f'  {sa_id} -> {sm_id} [style=dashed color="#aaaaaa" penwidth=1.0 '
                    f'label="α {mlp_ratios[l]:.2f}" '
                    f'constraint=false weight=0 arrowhead=open fontsize=7];'
                )
            else:
                lines.append(
                    f'  {sa_id} -> {sm_id} [style=bold color="{_RESID_COL}" penwidth=3.0 '
                    f'label="α {mlp_ratios[l]:.2f}" '
                    f'constraint=false weight=0 arrowhead=open fontsize=7];'
                )
            prev = sm_id
        else:
            prev = sa_id

        # Transition to gap node
        if gap_after is not None and l == gap_after and not gap_edge_done:
            lines.append(
                f'  {prev} -> GAP [style=dashed color="#bbbbbb" arrowhead=none];'
            )
            prev = "GAP"
            gap_edge_done = True

    lines += [
        "",
        '  OUT [label="Logits / Output" shape=box fillcolor="#fce4ec" '
        'color="#c62828" penwidth=1.5 fontcolor="#7f0000"];',
        f'  {prev} -> OUT [color="#555555" penwidth=1.5];',
        "",
        f'  // Stats: k={k_edges}  epsilon={epsilon:.5f}  coverage={mass_coverage:.0%}',
        "}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helper (SVG + PNG)
# ─────────────────────────────────────────────────────────────────────────────

def _render_dot(dot_str: str, out_stem: str) -> None:
    """
    Render DOT source to both SVG and PNG via subprocess with a hard timeout.
    Avoids the graphviz Python package, which calls dot with no timeout and
    hangs indefinitely on complex clustered graphs.
    """
    import subprocess, shutil

    dot_bin = shutil.which("dot")
    if not dot_bin:
        for candidate in ("/usr/local/bin/dot", "/usr/bin/dot",
                          "/opt/homebrew/bin/dot"):
            if Path(candidate).exists():
                dot_bin = candidate
                break
    if not dot_bin:
        print("  PNG/SVG skipped — dot binary not found. "
              "Install: sudo apt install graphviz  OR  brew install graphviz")
        return

    dot_path = Path(f"{out_stem}.dot")
    _RENDER_TIMEOUT = 90   # seconds per format (large models with many layers need more time)

    for fmt in ("svg", "png"):
        out = Path(f"{out_stem}.{fmt}")
        try:
            result = subprocess.run(
                [dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out)],
                capture_output=True, text=True, timeout=_RENDER_TIMEOUT,
            )
            if result.returncode == 0:
                print(f"  {fmt.upper():<5}  → {out}")
            else:
                print(f"  {fmt.upper()} failed: {result.stderr.strip()[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  {fmt.upper()} skipped — dot layout timed out "
                  f"(>{_RENDER_TIMEOUT}s). .dot and .md files are still valid.")
        except Exception as exc:
            print(f"  {fmt.upper()} error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Task suites
# ─────────────────────────────────────────────────────────────────────────────

TASK_SUITES: dict = {
    "quick": {
        "tasks": [
            "The cat sat on the mat.",
            "What is 7 times 8? The answer is",
            "Alice is the mother of Bob. Bob is the mother of Carol. "
            "Who is Alice's grandchild? The answer is",
        ],
        "labels": ["Simple sentence", "Arithmetic", "2-hop reasoning"],
    },

    "complexity_gradient": {
        "tasks": [
            # Lexical / surface
            "The dog barked loudly.",
            # Syntactic
            "The keys to the cabinet are on the table. The keys",
            # Single-hop factual
            "The capital of France is",
            # Arithmetic
            "17 plus 28 equals",
            # 2-hop compositional
            "Alice is the mother of Bob. Bob is the mother of Carol. "
            "Alice's grandchild is",
            # 3-hop compositional
            "Alice is the parent of Bob. Bob is the parent of Carol. "
            "Carol is the parent of Dana. Alice's great-grandchild is",
            # Logical syllogism
            "All mammals breathe air. Dolphins are mammals. "
            "Therefore, dolphins",
            # Analogy
            "Paris is to France as Berlin is to",
        ],
        "labels": [
            "Lexical",
            "Subject-verb agreement",
            "1-hop factual",
            "Arithmetic",
            "2-hop reasoning",
            "3-hop reasoning",
            "Logical syllogism",
            "Analogy",
        ],
    },

    "syntax": {
        "tasks": [
            "The cat sat on the mat.",
            "The keys to the cabinet are on the table. The keys",
            "The man who the dogs chased ran. The man",
            "She said that he believed that they would come. They",
            "Either the manager or the employees are responsible. They",
        ],
        "labels": [
            "Simple SVO",
            "Prepositional phrase attractor",
            "Relative clause (object-extracted)",
            "Long-range agreement (embedded clause)",
            "Either-or agreement",
        ],
    },

    "arithmetic": {
        "tasks": [
            "2 + 2 =",
            "17 + 28 =",
            "7 times 8 equals",
            "144 divided by 12 equals",
            "What is 15% of 200? The answer is",
            "If a train travels at 60 mph for 2.5 hours, it covers",
        ],
        "labels": [
            "Trivial addition",
            "2-digit addition",
            "Multiplication (single digit)",
            "Division",
            "Percentage",
            "Word problem",
        ],
    },

    "reasoning": {
        "tasks": [
            # 1-hop
            "Alice is the mother of Bob. Alice's child is",
            # 2-hop
            "Alice is the mother of Bob. Bob is the mother of Carol. "
            "Alice's grandchild is",
            # 3-hop
            "Alice is the parent of Bob. Bob is the parent of Carol. "
            "Carol is the parent of Dana. Alice's great-grandchild is",
            # Logical deduction
            "All birds have wings. A penguin is a bird. Therefore, a penguin has",
            # Negation + logic
            "No reptiles are warm-blooded. All mammals are warm-blooded. "
            "Therefore, snakes are",
            # Counterfactual
            "In a world where cats bark and dogs meow, if you hear barking "
            "outside you think it is a",
        ],
        "labels": [
            "1-hop chain",
            "2-hop chain",
            "3-hop chain",
            "Categorical syllogism",
            "Negation + deduction",
            "Counterfactual",
        ],
    },

    "world_knowledge": {
        "tasks": [
            "The capital of Japan is",
            "Shakespeare wrote the play Hamlet. The author of Hamlet is",
            "Water is made of hydrogen and",
            "The theory of relativity was developed by",
            "In 1969, Neil Armstrong became the first person to walk on the",
            "The largest planet in the solar system is",
        ],
        "labels": [
            "Capital city",
            "Author recall",
            "Chemical composition",
            "Scientific attribution",
            "Historical event",
            "Astronomy fact",
        ],
    },

    # Extends k-hop chains to depth 6 — tests how far the compute horizon retreats
    "deep_chains": {
        "tasks": [
            "Alice is the mother of Bob. Alice's child is",
            "Alice is the mother of Bob. Bob is the mother of Carol. "
            "Alice's grandchild is",
            "Alice is the parent of Bob. Bob is the parent of Carol. "
            "Carol is the parent of Dana. Alice's great-grandchild is",
            "Alice is the parent of Bob. Bob is the parent of Carol. "
            "Carol is the parent of Dana. Dana is the parent of Eve. "
            "Alice's great-great-grandchild is",
            "A is the parent of B. B is the parent of C. C is the parent of D. "
            "D is the parent of E. E is the parent of F. "
            "A's great-great-great-grandchild is",
            "A is the parent of B. B is the parent of C. C is the parent of D. "
            "D is the parent of E. E is the parent of F. F is the parent of G. "
            "A's descendant six generations down is",
        ],
        "labels": [
            "1-hop chain",
            "2-hop chain",
            "3-hop chain",
            "4-hop chain",
            "5-hop chain",
            "6-hop chain",
        ],
    },

    # Minimal-processing baseline — trivially predictable next tokens
    "surface": {
        "tasks": [
            "The sky is",
            "Water boils at one hundred degrees",
            "One two three four",
            "The dog barked at the",
            "Hello, my name",
            "The cat sat on the",
        ],
        "labels": [
            "Sky color (trivial)",
            "Boiling point (trivial)",
            "Number sequence",
            "Dog sentence (surface)",
            "Greeting (surface)",
            "Cat sentence (surface)",
        ],
    },

    # One representative task per capability type — broad cross-type survey
    "mixed": {
        "tasks": [
            "The dog barked loudly.",
            "The capital of France is",
            "2 + 2 =",
            "Alice is the mother of Bob. Alice's child is",
            "Alice is the mother of Bob. Bob is the mother of Carol. "
            "Alice's grandchild is",
            "Alice is the parent of Bob. Bob is the parent of Carol. "
            "Carol is the parent of Dana. Alice's great-grandchild is",
            "All birds have wings. A penguin is a bird. Therefore, a penguin has",
            "No reptiles are warm-blooded. All mammals are warm-blooded. "
            "Therefore, snakes are",
            "In a world where cats bark and dogs meow, if you hear barking "
            "outside you think it is a",
            "Paris is to France as Berlin is to",
        ],
        "labels": [
            "Surface",
            "Factual recall",
            "Trivial arithmetic",
            "1-hop chain",
            "2-hop chain",
            "3-hop chain",
            "Categorical syllogism",
            "Negation + deduction",
            "Counterfactual",
            "Analogy",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_task(
    model:          HookedTransformer,
    text:           str,
    label:          str,
    mass_coverage:  float,
    out_stem:       str,
    head_threshold: float,
    mag_threshold:  float = MAG_THRESHOLD,
    compact:        bool  = False,
    layer_window:   Optional[Tuple[int, int]] = None,
) -> None:
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    is_ao    = bool(getattr(model.cfg, "attn_only", False))
    model_name = getattr(model.cfg, "model_name", "unknown")

    tokens = model.to_tokens(text)
    if tokens.shape[-1] > 512:
        tokens = tokens[:, -512:]
    tokens = tokens.to(next(model.parameters()).device)

    print(f"  Computing per-head attribution scores …")
    head_scores, mlp_scores = compute_per_head_scores(model, tokens)

    # ── Attribution-based stats (k, ε) — kept for header display only ────────
    attn_layer_scores = head_scores.max(axis=1)   # [n_layers]
    _, _, eps, k = select_active_edges_by_mass_coverage(
        attn_layer_scores, mlp_scores, mass_fraction=mass_coverage,
    )

    # ── Magnitude-ratio criterion for block active / inactive ─────────────────
    # "An inactive skip block is when ||block_output|| / ||stream_input|| < threshold"
    print(f"  Computing magnitude ratios (threshold={mag_threshold:.3f}) …")
    attn_ratios, mlp_ratios = compute_magnitude_ratios(model, tokens)
    act_a = (attn_ratios > mag_threshold).tolist()
    act_m = (mlp_ratios  > mag_threshold).tolist()

    n_act_a = sum(act_a)
    n_act_m = sum(act_m)
    print(f"  k={k}  attn active={n_act_a}/{n_layers}  "
          f"mlp active={n_act_m}/{n_layers}  ε={eps:.5f}")
    print(f"  attn ratios: {' '.join(f'{r:.2f}' for r in attn_ratios)}")
    if not is_ao:
        print(f"  mlp  ratios: {' '.join(f'{r:.2f}' for r in mlp_ratios)}")

    common = dict(
        model_name=model_name,
        n_layers=n_layers,
        n_heads=n_heads,
        head_scores=head_scores,
        mlp_scores=mlp_scores,
        active_attn=act_a,
        active_mlp=act_m,
        attn_ratios=attn_ratios,
        mlp_ratios=mlp_ratios,
        task_text=text,
        task_label=label,
        mass_coverage=mass_coverage,
        epsilon=eps,
        k_edges=k,
        is_attn_only=is_ao,
        head_threshold=head_threshold,
        mag_threshold=mag_threshold,
    )

    # Auto-enable compact for large models (> 30 layers)
    emit_compact = compact or (n_layers > 30)
    # For very large models (> 40 layers), skip the full per-head graph to
    # avoid Graphviz timeouts; only the compact summary is practical.
    emit_full = n_layers <= 40

    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)

    # ── Mermaid (full per-head graph) ─────────────────────────────────────────
    if emit_full:
        md_path = Path(f"{out_stem}.md")
        mermaid_str = build_mermaid(**common)
        md_path.write_text(
            f"```mermaid\n{mermaid_str}\n```\n\n"
            f"<!-- Paste the block above at https://mermaid.live to render -->\n",
            encoding="utf-8",
        )
        print(f"  Mermaid  → {md_path}")

    # ── DOT — full per-head graph ─────────────────────────────────────────────
    if emit_full:
        dot_path = Path(f"{out_stem}.dot")
        dot_str  = build_dot(**common)
        dot_path.write_text(dot_str, encoding="utf-8")
        print(f"  DOT      → {dot_path}")
        _render_dot(dot_str, out_stem)

    # ── Compact DOT — layer-summary graph (no individual head nodes) ──────────
    if emit_compact:
        c_stem   = f"{out_stem}_compact"
        c_dot    = build_compact_dot(**common, layer_window=layer_window)
        c_path   = Path(f"{c_stem}.dot")
        c_path.write_text(c_dot, encoding="utf-8")
        print(f"  Compact  → {c_path}")
        _render_dot(c_dot, c_stem)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    suite_names = ", ".join(TASK_SUITES.keys())
    parser = argparse.ArgumentParser(
        description="Generate Mermaid.js / Graphviz / PNG computational graphs "
                    "of the active subgraph per task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Built-in suites: {suite_names}",
    )
    # ── Model(s) ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", default="gpt2",
        help="Single model (used when --models is not set).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Run across multiple models. Output dirs are named per model. "
             "Example: --models gpt2 gpt2-medium EleutherAI/pythia-160m",
    )
    # ── Task(s) ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--suite", default=None,
        choices=list(TASK_SUITES.keys()),
        help=f"Named task suite. Options: {suite_names}",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Custom task prompts (overrides --suite).",
    )
    parser.add_argument("--labels", nargs="+", default=None)
    # ── Options ───────────────────────────────────────────────────────────────
    parser.add_argument("--mass_coverage", type=float, default=0.90)
    parser.add_argument(
        "--head_threshold", type=float, default=0.15,
        help="Head active if score ≥ head_threshold × max-score in its layer.",
    )
    parser.add_argument(
        "--mag_threshold", type=float, default=MAG_THRESHOLD,
        help=(
            "Block active/inactive threshold (default %(default)s). "
            "A block is inactive (skip-dominant) when "
            "||block_output|| / ||stream_input|| < mag_threshold."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument(
        "--out", default="graphs/subgraph",
        help="Output path stem. With multiple models the model name is appended.",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help=(
            "Emit a compact layer-summary DOT graph (one Attn + one FFN box "
            "per layer, no individual head nodes). Auto-enabled for models "
            "with more than 30 layers."
        ),
    )
    parser.add_argument(
        "--max_layers", type=int, default=None,
        metavar="N",
        help=(
            "In compact mode, window the graph to the first N//2 and last N//2 "
            "layers with a '…' gap for the middle. Useful for 70B/80-layer models. "
            "Example: --max_layers 16 shows first 8 + last 8 layers."
        ),
    )
    parser.add_argument(
        "--list_presets", action="store_true",
        help="Print the table of large-model presets and exit.",
    )
    args = parser.parse_args()

    if args.list_presets:
        print("\nLarge-model presets (open-source, TransformerLens-compatible):\n")
        print(f"  {'Key':<18} {'HF repo':<50} {'L':>3} {'H':>3} "
              f"{'VRAM(4bit)':>10}  Notes")
        print("  " + "-" * 95)
        for key, (repo, nl, nh, vram, note) in LARGE_MODEL_PRESETS.items():
            print(f"  {key:<18} {repo:<50} {nl:>3} {nh:>3} "
                  f"{'~'+str(vram)+'GB':>10}  {note}")
        print()
        return

    # ── Resolve task list ─────────────────────────────────────────────────────
    if args.tasks:
        tasks  = args.tasks
        labels = args.labels or [f"Task {i+1}" for i in range(len(tasks))]
    elif args.suite:
        suite  = TASK_SUITES[args.suite]
        tasks  = suite["tasks"]
        labels = args.labels or suite["labels"]
    else:
        suite  = TASK_SUITES["quick"]
        tasks  = suite["tasks"]
        labels = suite["labels"]

    if len(labels) < len(tasks):
        labels += [f"Task {i+1}" for i in range(len(labels), len(tasks))]

    # ── Resolve model list ────────────────────────────────────────────────────
    model_names = args.models if args.models else [args.model]

    for model_name in model_names:
        safe = model_name.replace("/", "_").replace("-", "_")
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        try:
            model = load_model(model_name, device=args.device,
                               hf_token=args.hf_token)
            model.eval()
        except Exception as exc:
            print(f"  FAILED to load: {exc}")
            continue

        n_lay = model.cfg.n_layers
        n_hd  = model.cfg.n_heads
        arch  = ("parallel" if getattr(model.cfg, "parallel_attn_mlp", False)
                 else "attn-only" if getattr(model.cfg, "attn_only", False)
                 else "sequential")
        print(f"  n_layers={n_lay}  n_heads={n_hd}  arch={arch}")

        out_base = (f"{args.out}_{safe}"
                    if len(model_names) > 1 else args.out)

        # Layer window for compact graph
        layer_window: Optional[Tuple[int, int]] = None
        if args.max_layers is not None and args.max_layers < n_lay:
            half = args.max_layers // 2
            layer_window = (half, args.max_layers - half)
            print(f"  Layer window: first {half} + last {args.max_layers - half} "
                  f"of {n_lay} layers")

        for i, (text, label) in enumerate(zip(tasks, labels)):
            print(f"\n  [{i+1}/{len(tasks)}] {label!r}")
            print(f"    {text[:80]}{'...' if len(text) > 80 else ''}")
            stem = f"{out_base}_task{i}"
            try:
                process_task(
                    model=model,
                    text=text,
                    label=label,
                    mass_coverage=args.mass_coverage,
                    out_stem=stem,
                    head_threshold=args.head_threshold,
                    mag_threshold=args.mag_threshold,
                    compact=args.compact,
                    layer_window=layer_window,
                )
            except Exception as exc:
                print(f"  ERROR: {exc}")
                import traceback; traceback.print_exc()

        # free GPU memory between models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("Done.")
    print("→ View .md files at  https://mermaid.live")
    print("→ Open .png files directly in any image viewer")
    print("→ Manual render:  dot -Tpng file.dot -o file.png")


if __name__ == "__main__":
    main()
