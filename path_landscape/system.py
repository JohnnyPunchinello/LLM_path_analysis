"""System and Unit: the data structures that specify an information-flow system.

A System carries a directed multigraph over named units. Edges may be marked
recurrent (closing a loop in time). Units may carry a `parent` referring to a
larger unit at the next scale up; this defines the multiscale hierarchy.

Two derived static graphs are available:

  - `unroll(T)`   : time-unrolled DAG over T steps. Recurrent edges become
                    forward edges (u@t -> v@(t+1)); non-recurrent edges stay
                    within a step (u@t -> v@t). Time disappears into structure.
  - `coarsen()`   : the next-scale-up System, where each unit is collapsed into
                    its parent and within-parent edges are dropped.

Path enumeration runs on these static graphs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import networkx as nx


@dataclass
class Unit:
    """A basic computing unit.

    `scale`  : integer level in the hierarchy (0 = finest).
    `parent` : name of the unit one level up that contains this one
               (None at the top scale).
    `op`     : optional callable describing what this unit computes.
               Not used for path structure; available for downstream analysis.
    """

    name: str
    scale: int = 0
    parent: Optional[str] = None
    op: Optional[Callable] = None
    metadata: dict = field(default_factory=dict)


class System:
    """An information-flow system."""

    def __init__(self) -> None:
        self.units: dict[str, Unit] = {}
        # MultiDiGraph so a pair (u, v) can carry both a forward and a recurrent edge.
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.inputs: set[str] = set()
        self.outputs: set[str] = set()

    # ------------------------------------------------------------------ build

    def add_unit(
        self,
        name: str,
        scale: int = 0,
        parent: Optional[str] = None,
        op: Optional[Callable] = None,
        **metadata,
    ) -> Unit:
        if name in self.units:
            raise ValueError(f"unit already exists: {name}")
        u = Unit(name=name, scale=scale, parent=parent, op=op, metadata=metadata)
        self.units[name] = u
        self.graph.add_node(name, scale=scale, parent=parent)
        return u

    def add_edge(
        self,
        src: str,
        dst: str,
        weight: float = 1.0,
        recurrent: bool = False,
    ) -> None:
        if src not in self.units:
            raise ValueError(f"unknown source unit: {src}")
        if dst not in self.units:
            raise ValueError(f"unknown target unit: {dst}")
        self.graph.add_edge(src, dst, weight=weight, recurrent=recurrent)

    def set_input(self, *names: str) -> None:
        for n in names:
            if n not in self.units:
                raise ValueError(f"unknown input unit: {n}")
            self.inputs.add(n)

    def set_output(self, *names: str) -> None:
        for n in names:
            if n not in self.units:
                raise ValueError(f"unknown output unit: {n}")
            self.outputs.add(n)

    # --------------------------------------------------------------- unroll

    def unroll(self, T: int = 1) -> nx.DiGraph:
        """Time-unrolled DAG over T steps.

        Each unit `u` becomes T nodes named `f"{u}@{t}"` for t in 0..T-1.
        Non-recurrent edge (u, v) becomes (u@t -> v@t) for every t.
        Recurrent edge (u, v) becomes (u@t -> v@(t+1)) for t in 0..T-2.
        With T=1, recurrent edges have nowhere to go and are dropped.
        """
        if T < 1:
            raise ValueError("T must be >= 1")
        g = nx.DiGraph()
        for name, u in self.units.items():
            for t in range(T):
                g.add_node(
                    f"{name}@{t}",
                    base=name,
                    time=t,
                    scale=u.scale,
                    parent=u.parent,
                )
        for src, dst, data in self.graph.edges(data=True):
            recurrent = bool(data.get("recurrent", False))
            w = float(data.get("weight", 1.0))
            if recurrent:
                for t in range(T - 1):
                    g.add_edge(f"{src}@{t}", f"{dst}@{t + 1}", weight=w)
            else:
                for t in range(T):
                    g.add_edge(f"{src}@{t}", f"{dst}@{t}", weight=w)
        return g

    def unroll_sources(self, T: int = 1) -> list[str]:
        """Default sources for the unrolled graph: input units at t=0."""
        return [f"{n}@0" for n in self.inputs]

    def unroll_sinks(self, T: int = 1) -> list[str]:
        """Default sinks for the unrolled graph: output units at t=T-1."""
        return [f"{n}@{T - 1}" for n in self.outputs]

    # ------------------------------------------------------------- coarsen

    def coarsen(self) -> "System":
        """Collapse units into their parents to obtain the next-scale-up System.

        Units without a parent are kept as themselves (promoted one scale).
        Edges fully inside a parent are dropped (they live at the finer scale).
        Cross-parent edges are aggregated by summing their weights; an edge
        is recurrent at the macro scale iff any of its underlying micro edges
        are recurrent.
        """
        macro = System()

        # 1) macro units.
        for name, u in self.units.items():
            macro_name = u.parent if u.parent is not None else name
            if macro_name not in macro.units:
                macro_scale = (u.scale + 1) if u.parent is not None else u.scale
                macro.add_unit(macro_name, scale=macro_scale)

        # 2) macro edges (aggregate cross-parent edges).
        agg: dict[tuple[str, str], dict] = {}
        for src, dst, data in self.graph.edges(data=True):
            sm = self.units[src].parent or src
            dm = self.units[dst].parent or dst
            if sm == dm:
                continue
            entry = agg.setdefault((sm, dm), {"weight": 0.0, "recurrent": False})
            entry["weight"] += float(data.get("weight", 1.0))
            entry["recurrent"] = entry["recurrent"] or bool(data.get("recurrent", False))
        for (sm, dm), ed in agg.items():
            macro.add_edge(sm, dm, weight=ed["weight"], recurrent=ed["recurrent"])

        # 3) inputs / outputs lift to their parents.
        macro.set_input(*{(self.units[n].parent or n) for n in self.inputs})
        macro.set_output(*{(self.units[n].parent or n) for n in self.outputs})
        return macro

    # ------------------------------------------------------------- summary

    def summary(self) -> str:
        n_units = len(self.units)
        n_edges = self.graph.number_of_edges()
        n_recur = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("recurrent"))
        scales = sorted({u.scale for u in self.units.values()})
        return (
            f"System(units={n_units}, edges={n_edges} "
            f"[recurrent={n_recur}], inputs={len(self.inputs)}, "
            f"outputs={len(self.outputs)}, scales={scales})"
        )

    def __repr__(self) -> str:
        return self.summary()
