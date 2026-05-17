"""Side-by-side comparison: two architectures, both with skip / multi-exit
structure that produces *paths of varying lengths*, on the same conceptual
job (route information from inputs to outputs).

We work directly on the static path graph — no training. The point is that
the four cross-system metrics depend on the structure of the path landscape
itself, which is fully determined by the architecture once the system is
specified. If you have a trained PyTorch model you want to analyze, the
package provides:

    extract_mlp_flow / extract_rnn_flow  : weight edges by activation flow
    unit_activations / active_subgraph    : prune to nodes that actually fire
                                            on a test batch

so the same pipeline applies — but for the comparison story we don't need
training. The two systems below give meaningfully different landscapes
purely from their architecture.

Run:  python demo_compare_serious.py
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt

from path_landscape import (
    PathLandscape,
    System,
    enumerate_paths,
    sample_paths,
    compare,
    format_comparison,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")


# =================================================================== systems

def lm_like_system(depth: int = 8, width: int = 4) -> System:
    """LM-like: deep + skip-rich layered network with multi-exit readout.

    - Forward chain: every layer fully connects to the next.
    - 1-step skips at every layer (residual, weight 0.6).
    - 2-step skips every other layer (longer-range residual, weight 0.3).
    - Multi-exit: the last *two* layers are both designated outputs, so
      paths can terminate early or late, giving length variation.
    """
    sys = System()
    for layer in range(depth):
        for k in range(width):
            sys.add_unit(f"L{layer}_n{k}", scale=0,
                         parent=f"block{layer // 2}")

    # forward chain
    for layer in range(depth - 1):
        for i in range(width):
            for j in range(width):
                sys.add_edge(f"L{layer}_n{i}", f"L{layer + 1}_n{j}", weight=1.0)

    # short skip: layer -> layer + 2
    for layer in range(depth - 2):
        for k in range(width):
            sys.add_edge(f"L{layer}_n{k}", f"L{layer + 2}_n{k}", weight=0.6)

    # long skip: layer -> layer + 3 (sparser)
    for layer in range(depth - 3):
        sys.add_edge(f"L{layer}_n0", f"L{layer + 3}_n0", weight=0.35)

    sys.set_input(*[f"L0_n{k}" for k in range(width)])
    # multi-exit: outputs at depth-2 AND depth-1
    sys.set_output(
        *[f"L{depth - 2}_n{k}" for k in range(width)],
        *[f"L{depth - 1}_n{k}" for k in range(width)],
    )
    return sys


def neural_circuit_system(
    n_modules: int = 3,
    units_per_module: int = 4,
    T: int = 4,
) -> tuple[System, list[str]]:
    """Brain-like recurrent circuit. Returns (System, custom_sinks).

    - `n_modules` modules, each with within-module all-to-all recurrence.
    - Sparse cross-module forward and feedback edges.
    - Three readout units that each tap a *different* module, so paths to
      different sinks traverse different routes (and can have different
      lengths after unrolling).
    - Custom sinks let some readouts be reachable early in the unroll, others
      only late, giving variable path length to a single source.
    """
    sys = System()
    sys.add_unit("input", scale=0)
    for m in range(n_modules):
        for k in range(units_per_module):
            sys.add_unit(f"M{m}_n{k}", scale=0, parent=f"module{m}")
    for m in range(n_modules):
        sys.add_unit(f"readout_{m}", scale=0)

    # input fan-out (only into module 0; other modules are reached via
    # cross-module edges, which costs more time-steps)
    for k in range(units_per_module):
        sys.add_edge("input", f"M0_n{k}", weight=1.0)

    # within-module recurrence (all-to-all)
    for m in range(n_modules):
        for i in range(units_per_module):
            for j in range(units_per_module):
                sys.add_edge(f"M{m}_n{i}", f"M{m}_n{j}",
                             weight=0.7, recurrent=True)
    # cross-module forward edges (one per pair)
    for m in range(n_modules - 1):
        sys.add_edge(f"M{m}_n0", f"M{m + 1}_n0", weight=0.5)
    # cross-module feedback (recurrent)
    for m in range(1, n_modules):
        sys.add_edge(f"M{m}_n1", f"M{m - 1}_n1",
                     weight=0.4, recurrent=True)
    # readouts: each tapped by its own module
    for m in range(n_modules):
        for k in range(units_per_module):
            sys.add_edge(f"M{m}_n{k}", f"readout_{m}", weight=0.5)

    sys.set_input("input")
    # default System sink list is fine, but for path enumeration we use
    # custom sinks that read the readouts at *different* time steps,
    # producing paths of different length.
    sys.set_output(*[f"readout_{m}" for m in range(n_modules)])

    # readout_m at t = m + 1  (module 0 reachable at t=1, last module at t=n_modules)
    custom_sinks = [f"readout_{m}@{m + 1}" for m in range(n_modules)]
    return sys, custom_sinks


# =============================================================== run a model

def landscape(
    sys: System, T: int = 1, sinks: list[str] = None,
    n_samples: int = 1500, eps: float = 0.45, max_length: int = 32,
) -> PathLandscape:
    g = sys.unroll(T)
    sources = sys.unroll_sources(T)
    if sinks is None:
        sinks = sys.unroll_sinks(T)
    paths = sample_paths(g, sources, sinks,
                         n_samples=n_samples, max_length=max_length)
    if not paths:
        raise RuntimeError("no paths found from sources to sinks")
    L = PathLandscape(paths)
    L.cluster(eps=eps, min_samples=3)
    return L


# ======================================================================= main

def main():
    print("=" * 64)
    print("Building two systems (no training — purely structural)...")
    sys_lm = lm_like_system(depth=8, width=4)
    sys_circ, circ_sinks = neural_circuit_system(
        n_modules=3, units_per_module=4, T=4)
    print(f"  LM-like (deep + skip + multi-exit) : {sys_lm.summary()}")
    print(f"  circuit (modular + recurrent)      : {sys_circ.summary()}")
    print(f"  circuit custom sinks (multi-time)  : {circ_sinks}")

    print("\nSampling paths and clustering...")
    L_lm = landscape(sys_lm, T=1, n_samples=1500, eps=0.45)
    L_c  = landscape(sys_circ, T=4, sinks=circ_sinks,
                     n_samples=1500, eps=0.45)
    print(f"  LM-like : {L_lm.describe()}")
    print(f"  circuit : {L_c.describe()}")

    lengths_lm = [p.length for p in L_lm.paths]
    lengths_c  = [p.length for p in L_c.paths]
    print(f"  LM-like path-length range: {min(lengths_lm)} - {max(lengths_lm)}")
    print(f"  circuit path-length range: {min(lengths_c)} - {max(lengths_c)}")

    print("\nLandscape comparison")
    print("=" * 64)
    cmp = compare(L_lm, L_c, names=("LM-like", "circuit-like"))
    print(format_comparison(cmp))
    print("=" * 64)

    # ------------------------------------------------- visualize
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 1.0],
        hspace=0.55, wspace=0.22,
        top=0.95, bottom=0.05, left=0.06, right=0.97,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    L_lm.plot(ax=ax_a, show_legend=False,
              title=f"LM-like - landscape ({L_lm.n_modes} modes)")
    L_c.plot(ax=ax_b, show_legend=False,
             title=f"circuit-like - landscape ({L_c.n_modes} modes)")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    L_lm.plot_length_by_cluster(
        ax=ax_c, drop_noise=True, label_top_k=8,
        title="LM-like - paths by cluster (length on y)")
    L_c.plot_length_by_cluster(
        ax=ax_d, drop_noise=True, label_top_k=8,
        title="circuit-like - paths by cluster (length on y)")

    ax_e = fig.add_subplot(gs[2, :])
    ax_e.axis("off")
    table_text = format_comparison(cmp)
    ax_e.text(0.5, 0.95, "Comparison metrics", ha="center", va="top",
              fontsize=12, fontweight="bold")
    ax_e.text(0.5, 0.83, table_text, ha="center", va="top",
              family="monospace", fontsize=10)

    out = "demo_compare_serious.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
