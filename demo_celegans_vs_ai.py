"""Cross-substrate comparison: C. elegans-faithful synthetic chemotaxis
connectome vs. a matched AI policy network with chemotaxis-shaped I/O.

The biological side is a synthetic but C. elegans-faithful sensorimotor
circuit: real neuron names, hardwired chemotaxis projections from
Bargmann (2006), and stochastic surround-connectivity calibrated against
Varshney et al. 2011. The AI side is a deep + skip + multi-exit policy
network with the same input/output dimensionality (28 chemosensory-like
inputs, 60 motor-like outputs).

Both Systems are processed structurally -- no training, no activity
simulation. We sample paths from sensory/input units to motor/output
units, build path landscapes, and compare with the four cross-system
metrics. The point is to demonstrate the comparison pipeline on a
biologically-grounded case and read off which features converge and
which diverge.

Run:  python demo_celegans_vs_ai.py
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt

from path_landscape import (
    PathLandscape,
    sample_paths,
    compare,
    format_comparison,
)
from path_landscape.biological import (
    chemotaxis_circuit_celegans,
    ai_chemotaxis_agent,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")


def landscape(sys, T, n_samples=2000, eps=0.45, max_length=24, kernel=None):
    g = sys.unroll(T)
    paths = sample_paths(
        g, sys.unroll_sources(T), sys.unroll_sinks(T),
        n_samples=n_samples, max_length=max_length,
    )
    if not paths:
        raise RuntimeError("no paths found")
    L = PathLandscape(paths, kernel=kernel) if kernel else PathLandscape(paths)
    L.cluster(eps=eps, min_samples=3)
    return L


def main():
    print("=" * 64)
    print("Building systems (no training -- purely structural)...")
    bio = chemotaxis_circuit_celegans(seed=0)
    ai  = ai_chemotaxis_agent(
        n_input=len(bio.inputs),
        n_output=len(bio.outputs),
        depth=4, width=16, skip_every=2,
    )
    print(f"  C. elegans (synth.)  : {bio.summary()}")
    print(f"  AI agent (deep+skip) : {ai.summary()}")
    print(f"  matched I/O sizes    : sensory={len(bio.inputs)} (worm) "
          f"== input={len(ai.inputs)} (ai); "
          f"motor={len(bio.outputs)} (worm) == out={len(ai.outputs)} (ai)")

    # The two systems are ~150 nodes each. With dense fan-out (width 16+),
    # randomly-sampled paths share few intermediate units, so the default
    # eps would leave most paths as noise. We loosen eps and use a kernel
    # variant that emphasises ordered overlap over edge-Jaccard for these
    # high-fan-out systems (alpha=0.2 -> 80% sequence-similarity).
    from path_landscape import composite_similarity
    kernel = lambda p, q: composite_similarity(p, q, alpha=0.2)
    print("\nSampling paths and clustering (eps=0.55, alpha=0.2)...")
    L_bio = landscape(bio, T=3, n_samples=2500, eps=0.55,
                      max_length=18, kernel=kernel)
    L_ai  = landscape(ai,  T=1, n_samples=2500, eps=0.55,
                      max_length=18, kernel=kernel)
    print(f"  C. elegans : {L_bio.describe()}")
    print(f"  AI agent   : {L_ai.describe()}")

    bio_lengths = [p.length for p in L_bio.paths]
    ai_lengths  = [p.length for p in L_ai.paths]
    print(f"  C. elegans path-length range: {min(bio_lengths)} - {max(bio_lengths)}")
    print(f"  AI agent  path-length range: {min(ai_lengths)} - {max(ai_lengths)}")

    print("\nLandscape comparison")
    print("=" * 64)
    cmp = compare(L_bio, L_ai, names=("C. elegans", "AI agent"))
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
    L_bio.plot(ax=ax_a, show_legend=False,
               title=f"C. elegans (synth.) - landscape ({L_bio.n_modes} modes)")
    L_ai.plot(ax=ax_b, show_legend=False,
              title=f"AI agent - landscape ({L_ai.n_modes} modes)")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    L_bio.plot_length_by_cluster(
        ax=ax_c, drop_noise=True, label_top_k=8,
        title="C. elegans - paths by cluster (length on y)")
    L_ai.plot_length_by_cluster(
        ax=ax_d, drop_noise=True, label_top_k=8,
        title="AI agent - paths by cluster (length on y)")

    ax_e = fig.add_subplot(gs[2, :])
    ax_e.axis("off")
    table_text = format_comparison(cmp)
    ax_e.text(0.5, 0.95, "Comparison metrics", ha="center", va="top",
              fontsize=12, fontweight="bold")
    ax_e.text(0.5, 0.83, table_text, ha="center", va="top",
              family="monospace", fontsize=10)

    out = "demo_celegans_vs_ai.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
