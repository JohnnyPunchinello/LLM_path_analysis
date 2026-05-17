#!/usr/bin/env python3
"""
path_analyzer.py — Mechanistic Interpretability: Path Distribution Theory
==========================================================================

Implements PathAnalyzer which:
  1. Maps a HookedTransformer architecture to a block-level DAG (networkx)
  2. Counts paths by length via dynamic programming (Algorithm 1)
  3. Scores edges with Attribution Patching (AtP, gradient × activation)
  4. Derives empirical path distributions from the active subgraph G_active
  5. Computes H(π̂), E[L], and tail-mass ratio τ_k

Architecture support
--------------------
  Sequential (Pre-LN/RMS, e.g. Llama, GPT-2)
    • Each layer has TWO binary branch points: Attention, then MLP.
    • resid_pre → {skip, attn} → resid_mid → {skip, mlp} → resid_post
  Parallel (GPT-J style, parallel_attn_mlp=True)
    • Each layer has ONE ternary branch: skip + attn + mlp all from resid_pre.
    • resid_pre → {skip, attn, mlp} → resid_post
  Attention-only (attn_only=True)
    • Each layer has one binary branch: skip + attn.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathMetrics:
    """All path-distribution statistics for one (model, prompt) pair."""
    distribution:     np.ndarray   # π(l) — normalised, shape (max_path_len+1,)
    entropy:          float        # H(π) = −Σ π(l) log₂ π(l)  [bits]
    mean_path_length: float        # E[L] = Σ l·π(l)
    tail_mass_ratio:  float        # τ_k  = P(L > k) / P(L ≤ k)
    path_counts:      np.ndarray   # raw (unnormalised) counts
    max_path_len:     int
    cutoff_k:         int          # k used for τ_k


# ─────────────────────────────────────────────────────────────────────────────
# PathAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class PathAnalyzer:
    """
    Analyses path-length distributions in transformer models.

    Parameters
    ----------
    model : HookedTransformer
    """

    def __init__(self, model) -> None:
        self.model        = model
        self.cfg          = model.cfg
        self.n_layers     = self.cfg.n_layers
        self.is_parallel  = bool(getattr(self.cfg, "parallel_attn_mlp", False))
        self.is_attn_only = bool(getattr(self.cfg, "attn_only",         False))

        # Max possible path length depends on architecture:
        #   Parallel  (ternary branch: skip | attn | mlp) → max 1 compute per layer
        #   Attn-only (binary  branch: skip | attn)       → max 1 compute per layer
        #   Sequential(two binary branches: attn then mlp) → max 2 computes per layer
        if self.is_parallel or self.is_attn_only:
            self.max_path_len = self.n_layers
        else:
            self.max_path_len = 2 * self.n_layers

        self.dag = self._build_dag()

    # ──────────────────────────────────────────────────────────────────────
    # 1.  DAG construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_dag(self) -> nx.MultiDiGraph:
        """
        Build the block-level directed acyclic graph.

        Nodes — residual-stream checkpoints:
          'input'           : embedding output
          'resid_mid_{l}'   : after attention  (sequential only)
          'resid_post_{l}'  : after MLP / end of layer
          'output'          : before unembed

        Edges carry:
          type              : 'skip' | 'compute' | 'terminal'
          block             : 'attn' | 'mlp' | 'skip_attn' | 'skip_mlp' | 'skip' | 'final'
          layer             : int
          path_length_delta : 0 (skip) or 1 (compute)
        """
        G    = nx.MultiDiGraph()
        prev = "input"
        G.add_node(prev, kind="residual")

        for layer in range(self.n_layers):
            if self.is_parallel:
                # ── Parallel (GPT-J): ternary branch ──
                node = f"resid_post_{layer}"
                G.add_node(node, kind="residual", layer=layer)
                G.add_edge(prev, node, type="skip",    block="skip", layer=layer, path_length_delta=0)
                G.add_edge(prev, node, type="compute", block="attn", layer=layer, path_length_delta=1)
                if not self.is_attn_only:
                    G.add_edge(prev, node, type="compute", block="mlp", layer=layer, path_length_delta=1)
                prev = node

            elif self.is_attn_only:
                # ── Attn-only: binary branch (attn only) ──
                node = f"resid_post_{layer}"
                G.add_node(node, kind="residual", layer=layer)
                G.add_edge(prev, node, type="skip",    block="skip_attn", layer=layer, path_length_delta=0)
                G.add_edge(prev, node, type="compute", block="attn",      layer=layer, path_length_delta=1)
                prev = node

            else:
                # ── Sequential (Pre-LN/RMS, e.g. Llama): two binary branches ──
                mid  = f"resid_mid_{layer}"
                post = f"resid_post_{layer}"
                G.add_node(mid,  kind="residual", layer=layer, stage="mid")
                G.add_node(post, kind="residual", layer=layer, stage="post")
                # Attention branch
                G.add_edge(prev, mid,  type="skip",    block="skip_attn", layer=layer, path_length_delta=0)
                G.add_edge(prev, mid,  type="compute", block="attn",      layer=layer, path_length_delta=1)
                # MLP branch
                G.add_edge(mid,  post, type="skip",    block="skip_mlp",  layer=layer, path_length_delta=0)
                G.add_edge(mid,  post, type="compute", block="mlp",       layer=layer, path_length_delta=1)
                prev = post

        G.add_node("output", kind="output")
        G.add_edge(prev, "output", type="terminal", block="final", layer=-1, path_length_delta=0)
        return G

    # ──────────────────────────────────────────────────────────────────────
    # 2.  Algorithm 1 — dynamic-programming path counter
    # ──────────────────────────────────────────────────────────────────────

    def _path_count_dp(
        self,
        active_attn: List[bool],
        active_mlp:  Optional[List[bool]] = None,
    ) -> np.ndarray:
        """
        Algorithm 1 from Path Distribution Theory.

        Counts paths of every length l from input to output using DP.

        At each binary branch point:
            counts_new[l] = counts[l]         (skip, always active)
                          + counts[l-1]       (compute, if edge active)

        At each ternary branch point (parallel):
            counts_new[l] = counts[l]
                          + (has_attn) * counts[l-1]
                          + (has_mlp)  * counts[l-1]

        Parameters
        ----------
        active_attn : bool per layer — is the attention compute-edge active?
        active_mlp  : bool per layer — is the MLP compute-edge active?
                      (ignored for attn-only models)

        Returns
        -------
        counts : np.ndarray, shape (max_path_len + 1,)
                 counts[l] = number of input→output paths of length l
        """
        if active_mlp is None:
            active_mlp = [True] * self.n_layers

        counts = np.zeros(self.max_path_len + 1, dtype=np.float64)
        counts[0] = 1.0   # single path of length 0 at the input node

        for layer in range(self.n_layers):
            has_attn = bool(active_attn[layer])
            has_mlp  = bool(active_mlp[layer]) and not self.is_attn_only

            if self.is_parallel:
                # ── Ternary branch ──
                new = counts.copy()            # skip
                if has_attn:
                    new[1:] += counts[:-1]     # attn compute (+1)
                if has_mlp:
                    new[1:] += counts[:-1]     # mlp  compute (+1)
                counts = new

            else:
                # ── Attention binary branch ──
                new = counts.copy()            # skip
                if has_attn:
                    new[1:] += counts[:-1]
                counts = new

                # ── MLP binary branch (sequential only) ──
                if not self.is_attn_only:
                    new = counts.copy()        # skip
                    if has_mlp:
                        new[1:] += counts[:-1]
                    counts = new

        return counts

    # ──────────────────────────────────────────────────────────────────────
    # 3.  Analytical distribution (full architecture)
    # ──────────────────────────────────────────────────────────────────────

    def analytical_path_distribution(self) -> PathMetrics:
        """
        Path distribution when every computation edge is active.
        Depends only on architecture — no input data required.
        """
        all_active = [True] * self.n_layers
        counts     = self._path_count_dp(all_active, all_active)
        return _to_metrics(counts)

    # ──────────────────────────────────────────────────────────────────────
    # 4.  Attribution Patching (AtP) — edge scoring
    # ──────────────────────────────────────────────────────────────────────

    def compute_attribution_scores(
        self,
        tokens:              torch.Tensor,
        corrupted_tokens:    Optional[torch.Tensor] = None,
        target_pos:          int                    = -1,
        target_token_idx:    Optional[int]          = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Attribution Patching scores for every attention / MLP edge.

        Uses the first-order linear approximation of activation patching:

            AtP(e) = ∇_e [logit]_{baseline} · (a_e^{clean} − a_e^{baseline})

        With corrupted_tokens=None the baseline is the zero vector, which
        simplifies to:  AtP(e) ≈ |∇_e [logit]|·|a_e^{clean}|  (gradient × activation).

        Parameters
        ----------
        tokens            : LongTensor [1, seq_len]  clean input
        corrupted_tokens  : LongTensor [1, seq_len]  baseline (optional)
        target_pos        : sequence position for the target logit (−1 = last)
        target_token_idx  : vocabulary index of the target token;
                            uses model argmax if None

        Returns
        -------
        attn_scores : FloatTensor [n_layers]
        mlp_scores  : FloatTensor [n_layers]
        """
        # ── determine target token ──
        with torch.no_grad():
            logits_det = self.model(tokens, return_type="logits")
        if target_token_idx is None:
            target_token_idx = int(logits_det[0, target_pos].argmax())

        # ── clean activations ──
        clean_attn, clean_mlp = self._capture_activations(tokens)

        # ── baseline activations and gradients ──
        if corrupted_tokens is not None:
            base_attn, base_mlp      = self._capture_activations(corrupted_tokens)
            base_attn_g, base_mlp_g  = self._compute_gradients(
                corrupted_tokens, target_pos, target_token_idx
            )
        else:
            # Zero baseline: Δact = clean_act; gradient evaluated at clean point
            base_attn   = [torch.zeros_like(a) if a is not None else None for a in clean_attn]
            base_mlp    = [torch.zeros_like(a) if a is not None else None for a in clean_mlp]
            base_attn_g, base_mlp_g = self._compute_gradients(
                tokens, target_pos, target_token_idx
            )

        # ── AtP scores ──
        attn_scores = torch.zeros(self.n_layers)
        mlp_scores  = torch.zeros(self.n_layers)

        for l in range(self.n_layers):
            if (clean_attn[l] is not None
                    and base_attn[l]   is not None
                    and base_attn_g[l] is not None):
                delta          = (clean_attn[l] - base_attn[l]).to(base_attn_g[l].dtype)
                attn_scores[l] = float((base_attn_g[l] * delta).abs().mean())

            if (clean_mlp[l] is not None
                    and base_mlp[l]   is not None
                    and base_mlp_g[l] is not None):
                delta         = (clean_mlp[l] - base_mlp[l]).to(base_mlp_g[l].dtype)
                mlp_scores[l] = float((base_mlp_g[l] * delta).abs().mean())

        return attn_scores, mlp_scores

    # ──────────────────────────────────────────────────────────────────────
    # 5.  Empirical distribution (active subgraph)
    # ──────────────────────────────────────────────────────────────────────

    def empirical_path_distribution(
        self,
        tokens:               torch.Tensor,
        epsilon:              Optional[float] = None,
        corrupted_tokens:     Optional[torch.Tensor] = None,
        target_pos:           int                    = -1,
        target_token_idx:     Optional[int]          = None,
        epsilon_quantile:     float                  = 0.25,
    ) -> Tuple[PathMetrics, Dict[str, Any]]:
        """
        Compute empirical path distribution on the active subgraph G_active.

        Steps
        -----
        1. Compute AtP attribution scores for every edge.
        2. Threshold at ε → active_attn[l], active_mlp[l].
        3. Run Algorithm 1 on G_active.
        4. Return H(π̂), E[L], τ_k.

        Parameters
        ----------
        epsilon           : attribution threshold.  If None, set to the
                            `epsilon_quantile`-th quantile of all scores.
        epsilon_quantile  : quantile used to auto-set ε (default 0.25 → top 75 % active).

        Returns
        -------
        metrics : PathMetrics on the active subgraph
        info    : dict with scores, epsilon, active masks
        """
        attn_scores, mlp_scores = self.compute_attribution_scores(
            tokens, corrupted_tokens, target_pos, target_token_idx
        )

        all_scores = torch.cat([attn_scores, mlp_scores])
        if epsilon is None:
            epsilon = float(all_scores.quantile(epsilon_quantile))

        active_attn = (attn_scores > epsilon).tolist()
        active_mlp  = (mlp_scores  > epsilon).tolist()

        counts  = self._path_count_dp(active_attn, active_mlp)
        metrics = _to_metrics(counts)

        info: Dict[str, Any] = {
            "attn_scores":   attn_scores,
            "mlp_scores":    mlp_scores,
            "epsilon":       epsilon,
            "active_attn":   active_attn,
            "active_mlp":    active_mlp,
            "n_active_attn": sum(active_attn),
            "n_active_mlp":  sum(active_mlp),
        }
        return metrics, info

    # ──────────────────────────────────────────────────────────────────────
    # 6.  Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _capture_activations(
        self, tokens: torch.Tensor
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Return lists of detached attn_out / mlp_out activations, one per layer."""
        attn_acts: List[Optional[torch.Tensor]] = [None] * self.n_layers
        mlp_acts:  List[Optional[torch.Tensor]] = [None] * self.n_layers

        fwd_hooks = []
        for layer in range(self.n_layers):
            def _attn(act, hook, l=layer):
                attn_acts[l] = act.detach().clone()
                return act
            fwd_hooks.append((f"blocks.{layer}.hook_attn_out", _attn))
            if not self.is_attn_only:
                def _mlp(act, hook, l=layer):
                    mlp_acts[l] = act.detach().clone()
                    return act
                fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", _mlp))

        self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type=None)
        return attn_acts, mlp_acts

    def _compute_gradients(
        self,
        tokens:           torch.Tensor,
        target_pos:       int,
        target_token_idx: int,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """
        Compute ∂(target logit)/∂(attn_out_l) and ∂(target logit)/∂(mlp_out_l)
        for every layer l via reverse-mode autodiff.

        We anchor the computation graph at blocks.0.hook_resid_pre by detaching
        there and marking it requires_grad=True.  This avoids backpropagating
        through (potentially quantised) embedding weights while still giving
        correct gradients for all transformer blocks.
        """
        attn_acts: List[Optional[torch.Tensor]] = [None] * self.n_layers
        mlp_acts:  List[Optional[torch.Tensor]] = [None] * self.n_layers

        fwd_hooks: list = []

        # ── anchor: detach at residual-stream input, re-attach grad ──
        def _anchor(act, hook):
            # Creates a new leaf tensor; subsequent ops will have grad_fn.
            return act.detach().float().requires_grad_(True)

        fwd_hooks.append(("blocks.0.hook_resid_pre", _anchor))

        # ── capture intermediate activations with retain_grad ──
        for layer in range(self.n_layers):
            def _attn(act, hook, l=layer):
                if act.requires_grad:
                    act.retain_grad()
                attn_acts[l] = act
                return act
            fwd_hooks.append((f"blocks.{layer}.hook_attn_out", _attn))

            if not self.is_attn_only:
                def _mlp(act, hook, l=layer):
                    if act.requires_grad:
                        act.retain_grad()
                    mlp_acts[l] = act
                    return act
                fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", _mlp))

        try:
            with torch.enable_grad():
                logits = self.model.run_with_hooks(
                    tokens, fwd_hooks=fwd_hooks, return_type="logits"
                )
                target_logit = logits[0, target_pos, target_token_idx]
                target_logit.backward()
        except RuntimeError as exc:
            warnings.warn(f"Gradient computation failed: {exc}. Returning zero scores.")
            zeros_a = [None] * self.n_layers
            zeros_m = [None] * self.n_layers
            return zeros_a, zeros_m

        attn_grads = [
            (a.grad.detach().float() if a is not None and a.grad is not None else None)
            for a in attn_acts
        ]
        mlp_grads = [
            (a.grad.detach().float() if a is not None and a.grad is not None else None)
            for a in mlp_acts
        ]
        return attn_grads, mlp_grads

    # ──────────────────────────────────────────────────────────────────────
    # 7.  Architecture summary
    # ──────────────────────────────────────────────────────────────────────

    def architecture_summary(self) -> Dict[str, Any]:
        all_counts = self._path_count_dp([True] * self.n_layers, [True] * self.n_layers)
        arch_style = (
            "parallel (GPT-J)"           if self.is_parallel
            else "attn-only"             if self.is_attn_only
            else "sequential (Pre-LN/RMS)"
        )
        return {
            "model_name":    getattr(self.cfg, "model_name", "unknown"),
            "n_layers":      self.n_layers,
            "n_heads":       getattr(self.cfg, "n_heads",    None),
            "d_model":       getattr(self.cfg, "d_model",    None),
            "normalization": getattr(self.cfg, "normalization_type", "unknown"),
            "architecture":  arch_style,
            "max_path_len":  self.max_path_len,
            "total_paths":   int(all_counts.sum()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_metrics(counts: np.ndarray, cutoff_k: Optional[int] = None) -> PathMetrics:
    total = counts.sum()
    dist  = counts / total if total > 0.0 else counts.copy()

    # Entropy H(π)
    p = dist[dist > 0.0]
    H = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0

    # Mean path length E[L]
    lengths = np.arange(len(dist), dtype=np.float64)
    mu      = float(np.dot(lengths, dist))

    # Tail-mass ratio τ_k
    k      = cutoff_k if cutoff_k is not None else max(1, int(mu))
    short  = float(dist[:k + 1].sum())
    tail   = float(dist[k + 1:].sum())
    tau    = (tail / short) if short > 0.0 else float("inf")

    return PathMetrics(
        distribution=dist,
        entropy=H,
        mean_path_length=mu,
        tail_mass_ratio=tau,
        path_counts=counts,
        max_path_len=len(counts) - 1,
        cutoff_k=k,
    )


def select_active_edges_by_mass_coverage(
    attn_scores:   np.ndarray,
    mlp_scores:    np.ndarray,
    mass_fraction: float = 0.90,
) -> Tuple[List[bool], List[bool], float, int]:
    """
    Select the MINIMUM number of edges whose combined attribution scores
    account for >= mass_fraction of total attribution mass.

    Analogous to nucleus (top-p) sampling applied to edge importance:
    instead of fixing a percentage of edges, we fix a coverage target and let
    the attribution distribution determine how many edges are needed.

    Behaviour
    ---------
    Simple tokens  → attribution concentrated in a few dominant layers
                     → small k (sparse active subgraph)
    Complex tokens → attribution spread across many layers
                     → large k (dense active subgraph)

    Algorithm
    ---------
    1. Pool all 2L attribution scores.
    2. Sort descending; build cumulative sum.
    3. Find the smallest index i where cumsum[i] >= mass_fraction * total.
    4. epsilon = sorted_scores[i]  (the score of the i-th edge, 0-based).
    5. Mark every edge with score >= epsilon as active.
       (Ties at epsilon are included, so actual coverage >= mass_fraction.)

    Parameters
    ----------
    attn_scores   : float array [n_layers] — AtP magnitudes, attention edges
    mlp_scores    : float array [n_layers] — AtP magnitudes, MLP edges
    mass_fraction : float in (0, 1]       — coverage target (default 0.90)

    Returns
    -------
    active_attn : List[bool]  — per-layer attention edge active mask
    active_mlp  : List[bool]  — per-layer MLP edge active mask
    epsilon     : float       — threshold (k-th highest edge score)
    k_edges     : int         — number of edges selected
    """
    all_sc = np.concatenate([attn_scores, mlp_scores])
    total  = float(all_sc.sum())
    n      = len(all_sc)
    L      = len(attn_scores)

    # ── Edge case: zero attribution ──────────────────────────────────────────
    if total <= 0.0:
        # No signal; uniform prior — mark all edges active
        return [True] * L, [True] * L, 0.0, 2 * L

    # ── Cumulative coverage ──────────────────────────────────────────────────
    sorted_desc = np.sort(all_sc)[::-1]          # descending
    cumsum      = np.cumsum(sorted_desc)         # monotone increasing → 0..total
    target      = mass_fraction * total

    # Smallest 0-based index i where cumsum[i] >= target
    i       = int(np.searchsorted(cumsum, target, side="left"))
    i       = min(i, n - 1)                      # clamp to last valid index
    epsilon = float(sorted_desc[i])

    # ── Build masks ──────────────────────────────────────────────────────────
    # Use >= so that ties at epsilon are included
    # (can only increase coverage beyond mass_fraction)
    active_attn = (attn_scores >= epsilon).tolist()
    active_mlp  = (mlp_scores  >= epsilon).tolist()
    k_edges     = int(sum(active_attn) + sum(active_mlp))

    return active_attn, active_mlp, epsilon, k_edges


def path_entropy(distribution: np.ndarray) -> float:
    """H(π) = −Σ π(l) log₂ π(l)"""
    p = distribution[distribution > 0.0]
    return float(-np.sum(p * np.log2(p)))


def mean_path_length(distribution: np.ndarray) -> float:
    """E[L] = Σ l·π(l)"""
    return float(np.dot(np.arange(len(distribution), dtype=np.float64), distribution))


def tail_mass_ratio(distribution: np.ndarray, k: Optional[int] = None) -> float:
    """τ_k = P(L > k) / P(L ≤ k);  default k = floor(E[L])"""
    if k is None:
        k = max(1, int(mean_path_length(distribution)))
    short = float(distribution[:k + 1].sum())
    tail  = float(distribution[k + 1:].sum())
    return (tail / short) if short > 0.0 else float("inf")
