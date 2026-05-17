"""Small example systems used by the demo and as templates for new systems."""
from __future__ import annotations

from .system import System


def feedforward_chain(depth: int = 4, width: int = 3) -> System:
    """A layered feedforward network: `width` units per layer, full bipartite
    connections between consecutive layers. No skips, no recurrence."""
    sys = System()
    for layer in range(depth):
        for k in range(width):
            sys.add_unit(f"L{layer}_n{k}", scale=0)
    for layer in range(depth - 1):
        for i in range(width):
            for j in range(width):
                sys.add_edge(f"L{layer}_n{i}", f"L{layer + 1}_n{j}")
    sys.set_input(*[f"L0_n{k}" for k in range(width)])
    sys.set_output(*[f"L{depth - 1}_n{k}" for k in range(width)])
    return sys


def feedforward_with_skip(
    depth: int = 5, width: int = 3, skip_every: int = 2
) -> System:
    """Feedforward chain plus skip connections every `skip_every` layers."""
    sys = feedforward_chain(depth=depth, width=width)
    for layer in range(depth - skip_every):
        for k in range(width):
            sys.add_edge(f"L{layer}_n{k}", f"L{layer + skip_every}_n{k}")
    return sys


def simple_rnn(n_units: int = 3, n_input: int = 1, n_output: int = 1) -> System:
    """Recurrent system: an input, `n_units` recurrent units (all-to-all
    recurrence including self-loops), and an output. Recurrent edges are
    materialized only after `unroll(T)`."""
    sys = System()
    for k in range(n_input):
        sys.add_unit(f"in{k}", scale=0)
    for k in range(n_units):
        sys.add_unit(f"h{k}", scale=0)
    for k in range(n_output):
        sys.add_unit(f"out{k}", scale=0)
    for i in range(n_input):
        for k in range(n_units):
            sys.add_edge(f"in{i}", f"h{k}")
    for i in range(n_units):
        for j in range(n_units):
            sys.add_edge(f"h{i}", f"h{j}", recurrent=True)
    for k in range(n_units):
        for o in range(n_output):
            sys.add_edge(f"h{k}", f"out{o}")
    sys.set_input(*[f"in{k}" for k in range(n_input)])
    sys.set_output(*[f"out{k}" for k in range(n_output)])
    return sys


def mixture_of_experts(
    n_experts: int = 3, expert_depth: int = 3, expert_width: int = 1
) -> System:
    """An input fans out to `n_experts` parallel "expert" chains; each expert
    is a chain of `expert_depth` units; all experts converge on a single
    output. Designed to exhibit `n_experts` clearly separated path modes.
    """
    sys = System()
    sys.add_unit("in", scale=0)
    sys.add_unit("out", scale=0)
    for e in range(n_experts):
        # parent attaches each expert to a macro-scale "module".
        for d in range(expert_depth):
            for w in range(expert_width):
                sys.add_unit(f"E{e}_d{d}_w{w}", scale=0, parent=f"E{e}")
    for e in range(n_experts):
        for w in range(expert_width):
            sys.add_edge("in", f"E{e}_d0_w{w}")
        for d in range(expert_depth - 1):
            for wi in range(expert_width):
                for wj in range(expert_width):
                    sys.add_edge(f"E{e}_d{d}_w{wi}", f"E{e}_d{d+1}_w{wj}")
        for w in range(expert_width):
            sys.add_edge(f"E{e}_d{expert_depth-1}_w{w}", "out")
    sys.set_input("in")
    sys.set_output("out")
    return sys


def hierarchical_mlp(
    n_modules: int = 3, units_per_module: int = 3, depth: int = 3
) -> System:
    """A multi-scale MLP. Each layer is divided into `n_modules` modules of
    `units_per_module` units. Connections within a module are dense; between
    modules they are sparse (one cross-edge between aligned modules at
    consecutive layers). Each module across layers shares the same parent,
    so `coarsen()` produces a depth-`depth` chain of `n_modules` macro units.
    """
    sys = System()
    for layer in range(depth):
        for m in range(n_modules):
            for u in range(units_per_module):
                sys.add_unit(f"L{layer}_M{m}_n{u}", scale=0, parent=f"M{m}")
    # within-module dense connections between consecutive layers
    for layer in range(depth - 1):
        for m in range(n_modules):
            for i in range(units_per_module):
                for j in range(units_per_module):
                    sys.add_edge(
                        f"L{layer}_M{m}_n{i}", f"L{layer + 1}_M{m}_n{j}", weight=1.0
                    )
    # sparse cross-module edges (one per layer pair, m -> m+1)
    for layer in range(depth - 1):
        for m in range(n_modules - 1):
            sys.add_edge(
                f"L{layer}_M{m}_n0", f"L{layer + 1}_M{m + 1}_n0", weight=0.5
            )
    sys.set_input(*[f"L0_M{m}_n{u}" for m in range(n_modules)
                   for u in range(units_per_module)])
    sys.set_output(*[f"L{depth - 1}_M{m}_n{u}" for m in range(n_modules)
                    for u in range(units_per_module)])
    return sys


def two_module_network(units_per_module: int = 3) -> System:
    """Two modules (A, B), each a chain of `units_per_module` units,
    connected by a single cross-module edge from end-of-A to start-of-B.
    At the macro scale this collapses to a 2-node chain."""
    sys = System()
    for k in range(units_per_module):
        sys.add_unit(f"A{k}", scale=0, parent="A")
    for k in range(units_per_module):
        sys.add_unit(f"B{k}", scale=0, parent="B")
    for k in range(units_per_module - 1):
        sys.add_edge(f"A{k}", f"A{k + 1}")
        sys.add_edge(f"B{k}", f"B{k + 1}")
    sys.add_edge(f"A{units_per_module - 1}", "B0")
    sys.set_input("A0")
    sys.set_output(f"B{units_per_module - 1}")
    return sys
