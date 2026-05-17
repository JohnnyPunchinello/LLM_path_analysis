"""path_landscape: build the path representation of an information-flow system.

A *system* is a directed graph of computing units with optional recurrence and a
hierarchical (multiscale) structure. The package exposes:

  - System / Unit            : specify the system, its inputs/outputs, and scales.
  - Path / enumerate / sample: extract paths from input to output on the static
                               (time-unrolled, scale-coarsened) path graph.
  - similarity kernels       : compare paths.
  - PathLandscape            : the measured metric space of paths, with
                               clustering ("modes") and visualization.

Time and scale parameters shape the static path graph but do not appear in the
final representation: they are baked into the structure.
"""

from .system import System, Unit
from .paths import Path, enumerate_paths, sample_paths
from .similarity import (
    jaccard_nodes,
    jaccard_edges,
    ordered_overlap,
    composite_similarity,
    similarity_matrix,
)
from .landscape import PathLandscape, ClusterSummary
from .metrics import (
    n_modes,
    size_exponent,
    persistence_h0,
    persistence_h1,
    meta_graph,
    meta_graph_metrics,
    summarize,
    compare,
    format_comparison,
)

# `extract` requires torch and `agent` requires anthropic; import lazily so
# the rest of the package works without those.
def __getattr__(name):
    if name in {"extract_mlp_flow", "extract_rnn_flow", "prune_system"}:
        from . import extract as _extract
        return getattr(_extract, name)
    if name in {"analyze_emergence", "specify_system", "run_analysis"}:
        from . import agent as _agent
        return getattr(_agent, name)
    raise AttributeError(name)

__all__ = [
    "System",
    "Unit",
    "Path",
    "enumerate_paths",
    "sample_paths",
    "jaccard_nodes",
    "jaccard_edges",
    "ordered_overlap",
    "composite_similarity",
    "similarity_matrix",
    "PathLandscape",
    "ClusterSummary",
    "n_modes",
    "size_exponent",
    "persistence_h0",
    "persistence_h1",
    "meta_graph",
    "meta_graph_metrics",
    "summarize",
    "compare",
    "format_comparison",
]
