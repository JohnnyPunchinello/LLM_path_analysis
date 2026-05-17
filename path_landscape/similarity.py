"""Similarity kernels between paths.

Each kernel returns a value in [0, 1] (1 = identical, 0 = disjoint). The
default `composite_similarity` blends two intuitive signals:

  - **edge-Jaccard**  : *what components* the path uses.
  - **ordered overlap (LCS)** : *the order* in which it uses them.

Both are necessary: two paths through the same nodes but in opposite orders
should not be in the same cluster, but two paths sharing many sub-paths in
the same order should be.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .paths import Path


def jaccard_nodes(p: Path, q: Path) -> float:
    a, b = set(p.nodes), set(q.nodes)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def jaccard_edges(p: Path, q: Path) -> float:
    a, b = set(p.edges), set(q.edges)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def lcs_length(seq_a: tuple[str, ...], seq_b: tuple[str, ...]) -> int:
    """Longest common subsequence length (order-preserving)."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0
    prev = np.zeros(m + 1, dtype=np.int32)
    curr = np.zeros(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        a_i = seq_a[i - 1]
        for j in range(1, m + 1):
            if a_i == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return int(prev[m])


def ordered_overlap(p: Path, q: Path) -> float:
    L = lcs_length(p.nodes, q.nodes)
    denom = max(len(p.nodes), len(q.nodes))
    return L / denom if denom else 1.0


def composite_similarity(p: Path, q: Path, alpha: float = 0.5) -> float:
    """`alpha * edge-Jaccard + (1 - alpha) * ordered overlap`.

    With `alpha = 0.5` (default) the two signals are weighted equally.
    Use `alpha = 1` for compositional-only ("same parts"), `alpha = 0` for
    sequence-only ("same trajectory").
    """
    return alpha * jaccard_edges(p, q) + (1.0 - alpha) * ordered_overlap(p, q)


def similarity_matrix(
    paths: list[Path],
    kernel: Callable[[Path, Path], float] = composite_similarity,
) -> np.ndarray:
    """Symmetric pairwise similarity matrix."""
    n = len(paths)
    s = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            v = kernel(paths[i], paths[j])
            s[i, j] = s[j, i] = v
    return s
