"""Landscape comparison metrics.

Four scalar/structural summaries of a `PathLandscape`, designed for cross-system
comparison (e.g., trained LLM vs. biological circuit on the same task):

  - **n_modes**          : number of clusters (modes of computation).
  - **size_exponent**    : Zipf-style exponent of the cluster-size distribution.
                            Heavy tails -> large alpha; uniform sizes -> alpha~0.
  - **persistence (H0/H1)** : topological signature of the landscape.
                              H0 = mode lifespan under merging (single-linkage).
                              H1 = irreducible loops (Vietoris-Rips, requires
                              `ripser`; returns None if unavailable).
  - **meta_graph_metrics**: connectivity of the cluster meta-graph (nodes =
                            clusters, edges = shared intermediate units).
                            Tracks recombinability.

`compare(L1, L2)` runs all four on two landscapes and returns a dict suitable
for pretty-printing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import networkx as nx

from .landscape import PathLandscape


# ---------------------------------------------------------------- modes

def n_modes(L: PathLandscape) -> int:
    return L.n_modes


# --------------------------------------------------------- size exponent

def size_exponent(
    L: PathLandscape, min_clusters: int = 3
) -> tuple[float, float]:
    """Power-law exponent of the cluster-size distribution.

    Fits log(size_rank) = -alpha * log(rank) + const by least squares.
    Returns (alpha, R^2). If fewer than `min_clusters` clusters exist,
    returns (NaN, NaN).
    """
    sizes = np.array([c.size for c in L.cluster_summary()], dtype=float)
    if len(sizes) < min_clusters:
        return float("nan"), float("nan")
    sizes_sorted = np.sort(sizes)[::-1]
    ranks = np.arange(1, len(sizes_sorted) + 1, dtype=float)
    log_size = np.log(sizes_sorted)
    log_rank = np.log(ranks)
    slope, intercept = np.polyfit(log_rank, log_size, 1)
    pred = slope * log_rank + intercept
    ss_res = float(((log_size - pred) ** 2).sum())
    ss_tot = float(((log_size - log_size.mean()) ** 2).sum()) + 1e-12
    return float(-slope), float(1.0 - ss_res / ss_tot)


# ------------------------------------------------------------ persistence

def persistence_h0(L: PathLandscape, max_features: int = 20) -> list[float]:
    """Sorted (largest first) lifespans of dim-0 features (clusters under
    single-linkage merging on the path-distance matrix).

    The last feature has infinite persistence (the whole space) and is
    excluded.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    n = L.D.shape[0]
    if n < 2:
        return [0.0]
    Dcond = squareform(L.D, checks=False)
    Z = linkage(Dcond, method="single")
    heights = sorted(Z[:, 2].tolist(), reverse=True)
    # Drop the last (infinite) feature -- it's the whole connected component.
    return heights[:max_features]


def persistence_h1(
    L: PathLandscape, max_features: Optional[int] = None
) -> Optional[list[tuple[float, float]]]:
    """Persistence pairs (birth, death) of dim-1 features (loops/holes) via
    Vietoris-Rips. Requires the `ripser` package; returns None if it isn't
    installed.

    By default returns *all* features sorted by lifespan; pass `max_features`
    to truncate. (Earlier versions defaulted to 20, which was a misleading
    cap — comparisons appeared to "converge" at 20 because both systems
    saturated the cap.)
    """
    try:
        from ripser import ripser
    except ImportError:
        return None
    out = ripser(L.D, distance_matrix=True, maxdim=1)
    h1 = out["dgms"][1]
    pairs = [(float(b), float(d)) for b, d in h1]
    pairs.sort(key=lambda bd: -(bd[1] - bd[0]))  # by lifespan desc
    if max_features is not None:
        pairs = pairs[:max_features]
    return pairs


# ------------------------------------------------------------ meta-graph

def meta_graph(L: PathLandscape, min_overlap: int = 1) -> nx.Graph:
    """Cluster meta-graph: clusters are nodes; an edge connects two clusters
    sharing at least `min_overlap` *intermediate* units (excluding sources and
    sinks of the underlying paths). Edge weight = number of shared units.
    """
    summaries = L.cluster_summary()
    intermediate_nodes: dict[int, set[str]] = {}
    for c in summaries:
        idx = np.where(L.labels == c.label)[0]
        nodes: set[str] = set()
        for i in idx:
            p = L.paths[i]
            if len(p.nodes) > 2:
                nodes.update(p.nodes[1:-1])
        intermediate_nodes[c.label] = nodes

    g = nx.Graph()
    for c in summaries:
        g.add_node(c.label, size=c.size, weight=c.total_weight)
    labels = list(intermediate_nodes)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = intermediate_nodes[a] & intermediate_nodes[b]
            if len(shared) >= min_overlap:
                g.add_edge(a, b, weight=len(shared))
    return g


def meta_graph_metrics(L: PathLandscape) -> dict:
    """Summary of the meta-graph: number of clusters, giant-component fraction,
    mean degree, edge density."""
    g = meta_graph(L)
    n = g.number_of_nodes()
    if n == 0:
        return {"n_clusters": 0, "giant_fraction": 0.0,
                "mean_degree": 0.0, "density": 0.0}
    components = list(nx.connected_components(g))
    giant = max(len(c) for c in components)
    return {
        "n_clusters": n,
        "giant_fraction": giant / n,
        "mean_degree": (2.0 * g.number_of_edges() / n),
        "density": float(nx.density(g)),
    }


# --------------------------------------------------------------- compare

def summarize(L: PathLandscape) -> dict:
    """All four metrics for one landscape, packaged as a dict."""
    alpha, r2 = size_exponent(L)
    h0 = persistence_h0(L, max_features=10)
    h1 = persistence_h1(L)
    mg = meta_graph_metrics(L)
    return {
        "n_paths": len(L.paths),
        "n_modes": L.n_modes,
        "size_exponent": alpha,
        "size_exponent_r2": r2,
        "h0_top": h0[:5],
        "h0_max": max(h0) if h0 else 0.0,
        "h1_count": (len(h1) if h1 is not None else None),
        "h1_max_persistence": (
            max((d - b) for b, d in h1) if h1 else 0.0
        ),
        "meta": mg,
    }


def compare(
    L1: PathLandscape,
    L2: PathLandscape,
    names: tuple[str, str] = ("system A", "system B"),
) -> dict:
    """Run `summarize` on two landscapes; return {names[0]: ..., names[1]: ...}."""
    return {names[0]: summarize(L1), names[1]: summarize(L2)}


def format_comparison(comparison: dict) -> str:
    """Pretty-print the result of `compare`."""
    names = list(comparison)
    a, b = comparison[names[0]], comparison[names[1]]
    rows = [
        ("metric", names[0], names[1]),
        ("--------------------------", "--------", "--------"),
        ("n_paths",
         f"{a['n_paths']}", f"{b['n_paths']}"),
        ("n_modes",
         f"{a['n_modes']}", f"{b['n_modes']}"),
        ("size exponent (alpha)",
         f"{a['size_exponent']:.3f}", f"{b['size_exponent']:.3f}"),
        ("size exponent R^2",
         f"{a['size_exponent_r2']:.3f}", f"{b['size_exponent_r2']:.3f}"),
        ("H0 max persistence",
         f"{a['h0_max']:.3f}", f"{b['h0_max']:.3f}"),
        ("H1 features",
         (str(a['h1_count']) if a['h1_count'] is not None else "-"),
         (str(b['h1_count']) if b['h1_count'] is not None else "-")),
        ("H1 max persistence",
         f"{a['h1_max_persistence']:.3f}", f"{b['h1_max_persistence']:.3f}"),
        ("meta graph: clusters",
         f"{a['meta']['n_clusters']}", f"{b['meta']['n_clusters']}"),
        ("meta graph: giant fraction",
         f"{a['meta']['giant_fraction']:.3f}", f"{b['meta']['giant_fraction']:.3f}"),
        ("meta graph: mean degree",
         f"{a['meta']['mean_degree']:.3f}", f"{b['meta']['mean_degree']:.3f}"),
        ("meta graph: density",
         f"{a['meta']['density']:.3f}", f"{b['meta']['density']:.3f}"),
    ]
    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    col3 = max(len(r[2]) for r in rows)
    lines = [
        f"{r[0]:<{col1}}   {r[1]:>{col2}}   {r[2]:>{col3}}" for r in rows
    ]
    return "\n".join(lines)
