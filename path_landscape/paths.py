"""Paths on a static directed graph.

A `Path` is an ordered tuple of node names (the static graph already has time
and scale baked in). Two ways to obtain paths:

  - `enumerate_paths`: all simple paths from any source to any sink, capped.
  - `sample_paths`   : weighted random walks until a sink is hit (stochastic).

Path weight is the product of edge weights traversed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Path:
    """A directed path through a static path graph.

    `nodes`  : ordered tuple of node names (length L+1 for path of length L).
    `weight` : scalar contribution; default = product of edge weights.
    """

    nodes: tuple[str, ...]
    weight: float = 1.0

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple((self.nodes[i], self.nodes[i + 1]) for i in range(len(self.nodes) - 1))

    @property
    def length(self) -> int:
        return len(self.nodes) - 1

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:  # short, readable
        if len(self.nodes) <= 6:
            chain = " -> ".join(self.nodes)
        else:
            chain = (
                " -> ".join(self.nodes[:3]) + " ... " + " -> ".join(self.nodes[-2:])
            )
        return f"Path[{self.length}](w={self.weight:.3g}; {chain})"


def enumerate_paths(
    g: nx.DiGraph,
    sources: Iterable[str],
    sinks: Iterable[str],
    max_length: Optional[int] = None,
    max_paths: Optional[int] = 5000,
    weight_attr: str = "weight",
) -> list[Path]:
    """All simple paths from any source to any sink, capped at `max_paths`.

    Path weights are products of the per-edge `weight_attr` values. If the
    graph is large, set `max_length` and/or use `sample_paths` instead.
    """
    sources = [s for s in sources if s in g]
    sinks = [s for s in sinks if s in g]
    paths: list[Path] = []
    for src in sources:
        for sink in sinks:
            for nodes in nx.all_simple_paths(g, src, sink, cutoff=max_length):
                w = 1.0
                for i in range(len(nodes) - 1):
                    ed = g.get_edge_data(nodes[i], nodes[i + 1])
                    w *= float(ed.get(weight_attr, 1.0))
                paths.append(Path(nodes=tuple(nodes), weight=w))
                if max_paths is not None and len(paths) >= max_paths:
                    return paths
    return paths


def sample_paths(
    g: nx.DiGraph,
    sources: Iterable[str],
    sinks: Iterable[str],
    n_samples: int = 1000,
    max_length: int = 100,
    weight_attr: str = "weight",
    rng: Optional[np.random.Generator] = None,
) -> list[Path]:
    """Sample paths by weighted random walks from a uniformly chosen source
    until a sink is hit (or `max_length` steps elapse, in which case the walk
    is discarded). Useful when enumeration would explode.
    """
    rng = rng if rng is not None else np.random.default_rng()
    sources = [s for s in sources if s in g]
    sinks_set = set(sinks)
    if not sources:
        return []
    paths: list[Path] = []
    for _ in range(n_samples):
        nodes = [rng.choice(sources)]
        w = 1.0
        for _ in range(max_length):
            current = nodes[-1]
            if current in sinks_set:
                paths.append(Path(nodes=tuple(nodes), weight=w))
                break
            succ = list(g.successors(current))
            if not succ:
                break
            weights = np.array(
                [float(g[current][s].get(weight_attr, 1.0)) for s in succ],
                dtype=float,
            )
            total = weights.sum()
            if total <= 0:
                break
            choice = int(rng.choice(len(succ), p=weights / total))
            w *= float(weights[choice])
            nodes.append(succ[choice])
    return paths
