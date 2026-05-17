"""Train a small MLP, then build its functional path landscape.

Two landscapes are computed for the same trained network:

  (a) structural : edges weighted by |W|.
  (b) functional : edges weighted by mean |W * a_in| over the training batch
                   (the actual signal magnitude flowing along each edge).

Path weight = product of edge weights along the path. Paths are sampled (the
graph is dense) and clustered by the default composite kernel.

Usage:  python demo_torch.py
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt

from path_landscape import PathLandscape, sample_paths
from path_landscape.torch_bridge import (
    mlp_to_system,
    reweight_by_pathways,
    train_toy_mlp,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")


def landscape_from(sys, n_samples=400, eps=0.55, seed=0):
    g = sys.unroll(T=1)
    paths = sample_paths(
        g,
        sys.unroll_sources(T=1),
        sys.unroll_sinks(T=1),
        n_samples=n_samples,
        max_length=64,
    )
    L = PathLandscape(paths)
    L.cluster(eps=eps, min_samples=3)
    return L


def main():
    print("training small MLP (2 -> 8 -> 8 -> 3) on synthetic 3-class blobs...")
    model, X, y = train_toy_mlp(hidden=[8, 8], n_classes=3, n_input=2,
                                n_samples=600, epochs=300, seed=0)
    acc = (model(X).argmax(dim=1) == y).float().mean().item()
    print(f"  train accuracy = {acc:.3f}")

    sys_struct = mlp_to_system(model)
    print("structural system: ", sys_struct.summary())
    L_struct = landscape_from(sys_struct, n_samples=500, eps=0.55)
    print(f"  structural landscape: {L_struct.describe()}")

    sys_func = mlp_to_system(model)
    reweight_by_pathways(sys_func, model, X)
    print("functional system:  ", sys_func.summary())
    L_func = landscape_from(sys_func, n_samples=500, eps=0.55)
    print(f"  functional landscape: {L_func.describe()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    L_struct.plot(ax=axes[0], title="trained MLP — structural (|W|)")
    L_func.plot(ax=axes[1], title="trained MLP — functional (|W * a|)")
    plt.tight_layout()
    out = "demo_torch_landscape.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")

    print("\ntop 3 functional clusters (representative paths):")
    for c in L_func.cluster_summary()[:3]:
        print(f"  size={c.size}, weight={c.total_weight:.3g}")
        print(f"    {c.representative}")


if __name__ == "__main__":
    main()
