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
    ║   ╔═══ Multi-Head Attention ════════╗            ║
    ║   ║  [H0●] [H1○] [H2●] ... [Hn●]  ║            ║
    ║   ╚═════════════════════════════════╝            ║
    ║                      ↓                           ║
    ║                    (Σα) ←─ ─ ─ (residual)        ║
    ║                      ↓                           ║
    ║                   [FFN]                          ║
    ║                      ↓                           ║
    ║                    (Σα) ←─ ─ ─ (residual)        ║
    ╚══════════════════════════════════════════════════╝
         |
    ╔══ Layer 1 ══════════════════════════════════════╗
    ...

  Active head / block  : coloured fill + bold border
  Inactive             : grey
  Residual skip path   : dashed arrow labelled α
  Summation node       : circle labelled Σα

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

def _is_llama_family(name: str) -> bool:
    return any(k in name.lower() for k in ("llama", "mistral", "gemma", "falcon"))


def load_model(model_name: str, device: str = "cuda",
               hf_token: Optional[str] = None) -> HookedTransformer:
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
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = HookedTransformer.from_pretrained(
                model_name, quantization_config=bnb,
                device_map="auto", **extra)
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
    active_attn:  List[bool],      # [n_layers]  layer-level
    active_mlp:   List[bool],      # [n_layers]
    task_text:    str,
    task_label:   str,
    mass_coverage: float,
    epsilon:      float,
    k_edges:      int,
    is_attn_only: bool = False,
    head_threshold: float = 0.15,
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

        sa_id    = _mermaid_node_id("SA", l)     # Σα after attn
        sm_id    = _mermaid_node_id("SM", l)     # Σα after MLP
        ffn_id   = _mermaid_node_id("FFN", l)

        # Layer summary for subgraph label
        n_act_h   = sum(act_h) if active_attn[l] else 0
        head_info = f"{n_act_h}/{n_heads} heads" if not is_attn_only else ""
        mlp_info  = "FFN ✓" if active_mlp[l] else "FFN ✗"
        sg_label  = (f"Layer {l}  |  Attn: {head_info}  {mlp_info}"
                     if not is_attn_only else f"Layer {l}  |  {head_info}")

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

        # Σα after attention
        lines.append(f"    {sa_id}((\"Σα\")):::sigma")

        # MLP / FFN
        if not is_attn_only:
            m_norm = float(mlp_scores[l]) / ms_max
            if active_mlp[l]:
                lines.append(f"    {ffn_id}[\"FFN  L{l}\"]:::active_mlp")
            else:
                lines.append(f"    {ffn_id}[\"FFN  L{l}\"]:::inactive_mlp")
            # Σα after MLP
            lines.append(f"    {sm_id}((\"Σα\")):::sigma")

        lines.append("  end")
        lines.append("")

        # ── Edges into this layer ─────────────────────────────────────────
        # Residual skip: prev_sum --dashed--> Σα_attn  (always)
        lines.append(f"  {prev_sum} -. α .-> {sa_id}")

        for h, hid in enumerate(head_ids):
            if active_attn[l] and act_h[h]:
                lines.append(f"  {prev_sum} --> {hid}")
                lines.append(f"  {hid} --> {sa_id}")
            # inactive heads: draw thin edge to Σα (no arrow through head)

        if not is_attn_only:
            # Residual skip: Σα_attn --dashed--> Σα_mlp
            lines.append(f"  {sa_id} -. α .-> {sm_id}")
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
    active_attn:  List[bool],
    active_mlp:   List[bool],
    task_text:    str,
    task_label:   str,
    mass_coverage: float,
    epsilon:      float,
    k_edges:      int,
    is_attn_only: bool = False,
    head_threshold: float = 0.15,
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
        '  splines=line;',             # fast layout; curved stalls on large cluster graphs
        '  compound=true;',            # needed for lhead/ltail cluster clipping
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
        '    LEG_R  [label="Residual stream \\ncheckpoint", shape=diamond,'
        f'            fillcolor="{_RESID_NODE}", style=filled, fontcolor="#1a5276", fontsize=7];',
        '    { rank=same; LEG_A; LEG_AI; LEG_F; LEG_FI; LEG_R; }',
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
        sa_id  = f"SA_{l}"
        sm_id  = f"SM_{l}"
        ffn_id = f"FFN_{l}"
        rs_in  = f"RS_{l}"       # stream checkpoint entering this layer
        rs_out = f"RS_{l+1}"     # stream checkpoint leaving this layer

        sg_label = (f"Layer {l}  |  {n_ah}/{n_heads} heads"
                    + ("  FFN \u2713" if active_mlp[l] else "  FFN \u2717")
                    if not is_attn_only
                    else f"Layer {l}  |  {n_ah}/{n_heads} heads")

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

        # Σα after attention
        lines.append("    " + dot_node(
            sa_id, "\u03a3", shape="circle",
            fillcolor=_SIGMA_BG, color="#444444",
            penwidth=1.5, fontcolor="#333333",
            width="0.35", height="0.35", fixedsize="true"))

        if not is_attn_only:
            m_norm = float(mlp_scores[l]) / ms_max
            if active_mlp[l]:
                fc_m = _mlp_colour(m_norm)
                lines.append("    " + dot_node(
                    ffn_id, f"FFN  L{l}", shape="box",
                    fillcolor=fc_m, color="#1a5276",
                    penwidth=2.0, fontcolor="#ffffff"))
            else:
                lines.append("    " + dot_node(
                    ffn_id, f"FFN  L{l}", shape="box",
                    fillcolor=_INACTIVE, color="#cccccc",
                    penwidth=0.8, fontcolor="#aaaaaa"))

            lines.append("    " + dot_node(
                sm_id, "\u03a3", shape="circle",
                fillcolor=_SIGMA_BG, color="#444444",
                penwidth=1.5, fontcolor="#333333",
                width="0.35", height="0.35", fixedsize="true"))

        lines.append("  }")   # end cluster
        lines.append("")

        # ── Compute edges (attn, MLP) ──────────────────────────────────────
        for h in range(n_heads):
            hid = f"H_{l}_{h}"
            if active_attn[l] and act_h[h]:
                lines.append(dot_edge(rs_in, hid,
                                       color="#d62728", penwidth=1.6,
                                       weight="2"))
                lines.append(dot_edge(hid, sa_id,
                                       color="#d62728", penwidth=1.6,
                                       weight="2"))

        if not is_attn_only:
            if active_mlp[l]:
                lines.append(dot_edge(sa_id, ffn_id,
                                       color="#1f77b4", penwidth=1.8,
                                       weight="2"))
                lines.append(dot_edge(ffn_id, sm_id,
                                       color="#1f77b4", penwidth=1.8,
                                       weight="2"))

        # ── Residual connections (teal, curved, constraint=false) ──────────
        # These bypass the cluster boxes as clearly visible teal arcs.
        # Skip around attention: RS_in ──teal──> Σ_attn
        lines.append(dot_edge(rs_in, sa_id,
                               style="bold", color=_RESID_COL,
                               penwidth=3.5, label="\u03b1",
                               constraint=False, weight="0",
                               arrowhead="open"))

        if not is_attn_only:
            # Skip around MLP: Σ_attn ──teal──> Σ_mlp
            lines.append(dot_edge(sa_id, sm_id,
                                   style="bold", color=_RESID_COL,
                                   penwidth=3.5, label="\u03b1",
                                   constraint=False, weight="0",
                                   arrowhead="open"))
            # Backbone: Σ_mlp ──teal──> RS_out (next checkpoint)
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
    _RENDER_TIMEOUT = 30   # seconds per format; increase if you have a very large model

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
    model:         HookedTransformer,
    text:          str,
    label:         str,
    mass_coverage: float,
    out_stem:      str,
    head_threshold: float,
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

    # Layer-level active mask via mass-coverage rule
    attn_layer_scores = head_scores.max(axis=1)   # [n_layers]
    act_a, act_m, eps, k = select_active_edges_by_mass_coverage(
        attn_layer_scores, mlp_scores, mass_fraction=mass_coverage,
    )

    n_act_a = sum(act_a)
    n_act_m = sum(act_m)
    print(f"  Layer-level: k={k}  attn={n_act_a}/{n_layers}  "
          f"mlp={n_act_m}/{n_layers}  ε={eps:.5f}")

    common = dict(
        model_name=model_name,
        n_layers=n_layers,
        n_heads=n_heads,
        head_scores=head_scores,
        mlp_scores=mlp_scores,
        active_attn=act_a,
        active_mlp=act_m,
        task_text=text,
        task_label=label,
        mass_coverage=mass_coverage,
        epsilon=eps,
        k_edges=k,
        is_attn_only=is_ao,
        head_threshold=head_threshold,
    )

    # ── Mermaid ───────────────────────────────────────────────────────────────
    md_path = Path(f"{out_stem}.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid_str = build_mermaid(**common)
    md_path.write_text(
        f"```mermaid\n{mermaid_str}\n```\n\n"
        f"<!-- Paste the block above at https://mermaid.live to render -->\n",
        encoding="utf-8",
    )
    print(f"  Mermaid  → {md_path}")

    # ── DOT ───────────────────────────────────────────────────────────────────
    dot_path = Path(f"{out_stem}.dot")
    dot_str  = build_dot(**common)
    dot_path.write_text(dot_str, encoding="utf-8")
    print(f"  DOT      → {dot_path}")

    # ── SVG + PNG rendering ───────────────────────────────────────────────────
    _render_dot(dot_str, out_stem)


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument(
        "--out", default="graphs/subgraph",
        help="Output path stem. With multiple models the model name is appended.",
    )
    args = parser.parse_args()

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
