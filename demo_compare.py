"""Side-by-side comparison of two systems on the same task.

Demonstrates the four landscape metrics intended for cross-system claims
(e.g., trained LLM vs. biological circuit on a matched task):

  1. number of modes
  2. cluster-size exponent (power-law fit)
  3. persistence diagram, dim-0 and dim-1
  4. meta-graph connectivity (recombinability)

We compare two toy archetypes: a "transformer-like" deep + skip feedforward
network, and a "brain-like" recurrent modular network. Both attempt the same
abstract job: route information from inputs to outputs. The metrics expose
which features of the landscape are conserved across the two architectures
and which are substrate-specific.

Run:  python demo_compare.py
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from path_landscape import (
    PathLandscape,
    System,
    enumerate_paths,
    sample_paths,
    compare,
    format_comparison,
    persistence_h0,
    persistence_h1,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")


# --------------------------------------------------------- system builders

def transformer_like(depth: int = 6, width: int = 4, skip_every: int = 2) -> System:
    """Deep, narrow feedforward network with skip connections every
    `skip_every` layers. Stand-in for a transformer-style stack."""
    sys = System()
    for layer in range(depth):
        for k in range(width):
            sys.add_unit(f"L{layer}_n{k}", scale=0,
                         parent=f"block{layer // 2}")
    for layer in range(depth - 1):
        for i in range(width):
            for j in range(width):
                sys.add_edge(f"L{layer}_n{i}", f"L{layer + 1}_n{j}")
    for layer in range(depth - skip_every):
        for k in range(width):
            sys.add_edge(f"L{layer}_n{k}", f"L{layer + skip_every}_n{k}",
                         weight=0.6)
    sys.set_input(*[f"L0_n{k}" for k in range(width)])
    sys.set_output(*[f"L{depth - 1}_n{k}" for k in range(width)])
    return sys


def brain_like(
    n_modules: int = 3, units_per_module: int = 3, T: int = 4
) -> System:
    """Modular recurrent system. `n_modules` modules with within-module
    all-to-all recurrent connectivity, sparse cross-module forward edges,
    plus an input fan-out and a small readout. Time unrolling provides the
    long forward paths."""
    sys = System()
    sys.add_unit("in", scale=0)
    for m in range(n_modules):
        for k in range(units_per_module):
            sys.add_unit(f"M{m}_n{k}", scale=0, parent=f"module{m}")
    sys.add_unit("out", scale=0)

    # input fan-out
    for m in range(n_modules):
        for k in range(units_per_module):
            sys.add_edge("in", f"M{m}_n{k}")
    # within-module recurrence (all-to-all)
    for m in range(n_modules):
        for i in range(units_per_module):
            for j in range(units_per_module):
                sys.add_edge(f"M{m}_n{i}", f"M{m}_n{j}",
                             weight=0.8, recurrent=True)
    # sparse cross-module forward edges
    for m in range(n_modules - 1):
        sys.add_edge(f"M{m}_n0", f"M{m + 1}_n0", weight=0.5)
        sys.add_edge(f"M{m + 1}_n0", f"M{m}_n1",
                     weight=0.3, recurrent=True)  # feedback
    # readout
    for m in range(n_modules):
        for k in range(units_per_module):
            sys.add_edge(f"M{m}_n{k}", "out", weight=0.5)
    sys.set_input("in")
    sys.set_output("out")
    return sys


def landscape_for(
    sys: System, T: int = 1, n_samples: int = 1500, eps: float = 0.45
) -> PathLandscape:
    g = sys.unroll(T)
    paths = sample_paths(
        g,
        sys.unroll_sources(T),
        sys.unroll_sinks(T),
        n_samples=n_samples,
        max_length=64,
    )
    if not paths:
        raise RuntimeError("no paths found")
    L = PathLandscape(paths)
    L.cluster(eps=eps, min_samples=3)
    return L


# ------------------------------------------------------------ visualization

def plot_persistence(L: PathLandscape, ax, title: str) -> None:
    h0 = persistence_h0(L, max_features=30)
    h1 = persistence_h1(L)
    if h0:
        for h in h0[:25]:
            ax.plot([0.0, h], [h, h], color="#0c1220", linewidth=0.7,
                    alpha=0.6)
        ax.scatter([0.0] * len(h0[:25]), h0[:25], color="#7dc8dc",
                   s=18, zorder=4, label=f"H0 (top 25 of {len(h0)})")
    if h1:
        births = [b for b, _ in h1]
        deaths = [d for b, d in h1]
        ax.scatter(births, deaths, color="#c83c37", s=24, zorder=5,
                   label=f"H1 ({len(h1)})", edgecolors="#0c1220",
                   linewidths=0.4)
    lim = 1.0
    if h0:
        lim = max(lim, max(h0))
    if h1:
        lim = max(lim, max(d for _, d in h1) if h1 else 0)
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.4, alpha=0.4)
    ax.set_xlim(0, lim * 1.05); ax.set_ylim(0, lim * 1.05)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=7, frameon=False)


# ----------------------------------------------------------------- main

def main():
    print("Building two systems...")
    sysA = transformer_like(depth=6, width=4, skip_every=2)
    sysB = brain_like(n_modules=3, units_per_module=3)
    print(f"  transformer-like : {sysA.summary()}")
    print(f"  brain-like       : {sysB.summary()}")

    print("\nComputing landscapes (this may take a moment)...")
    LA = landscape_for(sysA, T=1, n_samples=800, eps=0.45)
    LB = landscape_for(sysB, T=4, n_samples=800, eps=0.50)
    print(f"  transformer-like : {LA.describe()}")
    print(f"  brain-like       : {LB.describe()}")

    print("\nLandscape comparison")
    print("=" * 60)
    cmp = compare(LA, LB, names=("transformer-like", "brain-like"))
    print(format_comparison(cmp))
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    LA.plot(ax=axes[0, 0], title="transformer-like — landscape")
    LB.plot(ax=axes[0, 1], title="brain-like — landscape")
    plot_persistence(LA, axes[1, 0], "transformer-like — persistence")
    plot_persistence(LB, axes[1, 1], "brain-like — persistence")
    plt.tight_layout()
    out = "demo_compare.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
