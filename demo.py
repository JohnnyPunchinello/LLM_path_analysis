"""Run the path-landscape pipeline on a few example systems and save a figure.

Usage:  python demo.py

Compares:
  (a) plain feedforward chain         -> few modes, all paths the same length
  (b) feedforward + skip connections  -> more modes (length diversity)
  (c) recurrent network unrolled T=4  -> modes by recurrence depth
  (d) two-module net micro vs macro   -> illustrates scale coarsening
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt

from path_landscape import PathLandscape, enumerate_paths
from path_landscape.examples import (
    feedforward_chain,
    feedforward_with_skip,
    mixture_of_experts,
    simple_rnn,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")


def landscape_for(sys, T=1, max_paths=5000, eps=0.4, max_length=None):
    g = sys.unroll(T)
    paths = enumerate_paths(
        g,
        sys.unroll_sources(T),
        sys.unroll_sinks(T),
        max_paths=max_paths,
        max_length=max_length,
    )
    if not paths:
        raise RuntimeError("no paths found from inputs to outputs")
    L = PathLandscape(paths)
    L.cluster(eps=eps, min_samples=2)
    return L


def report(name, L):
    print(f"\n=== {name} ===")
    print(f"  {L.describe()}")
    for c in L.cluster_summary():
        print(f"  {c}")


def main():
    # Two columns: MDS scatter (left) and length-x-cluster bar plot (right).
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    rows = [
        ("feedforward chain", feedforward_chain(depth=5, width=3),
         dict(T=1, max_paths=5000, eps=0.45)),
        ("feedforward + skip (depth=8, skip=3)",
         feedforward_with_skip(depth=8, width=3, skip_every=3),
         dict(T=1, max_paths=8000, eps=0.45)),
        ("RNN unrolled (T=4)", simple_rnn(n_units=3, n_input=1, n_output=1),
         dict(T=4, max_paths=4000, eps=0.5)),
        ("mixture of experts",
         mixture_of_experts(n_experts=3, expert_depth=3, expert_width=2),
         dict(T=1, max_paths=5000, eps=0.55)),
    ]
    for r, (name, sys, kw) in enumerate(rows):
        print(sys.summary())
        L = landscape_for(sys, **kw)
        report(name, L)
        L.plot(ax=axes[r, 0], show_legend=False, title=f"{name} - landscape")
        L.plot_length_by_cluster(
            ax=axes[r, 1], drop_noise=True, label_top_k=8,
            title=f"{name} - paths by cluster (length on y)",
        )

    plt.tight_layout()
    out = "demo_landscape.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
