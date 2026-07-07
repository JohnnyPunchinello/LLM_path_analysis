#!/usr/bin/env python3
"""
distribution_analysis.py
========================

Tests the "distribution idea": do prompts of the SAME category invoke the SAME
set of active attention heads?  Consumes a dead-head JSON produced by
`dead_heads_distribution.py` on the `consistency` suite (one entry per prompt,
each carrying a per-head score matrix and a label "{category}#{i}").

For each category it computes:

  activation-frequency map   f_c(l,h) = P(head (l,h) active | prompt in c)
                             — the core object; its histogram is the "distribution".
  within-category overlap    mean pairwise Jaccard of active-head sets (binary).
  between-category overlap    cross-category Jaccard (should be << within).
  rank consistency           per-layer Spearman of head scores across prompts,
                             averaged over layers — a THRESHOLD-FREE consistency
                             measure (removes the active/inactive cutoff entirely).
  variance decomposition     per-head eta^2 = between-category / total variance
                             of activity across all prompts — how strongly task
                             category determines whether a head fires.
  permutation null           within-Jaccard z-score vs. label-shuffled masks.
  core set + geometry        heads with f_c >= theta; its depth-span and
                             heads-per-layer (to line up against a category's
                             serial depth D / parallel width W).

Outputs a printed summary, a report.json, and three figures:
  fig_freq_hist.png   trimodal f_c histograms per category
  fig_jaccard.png     within/between Jaccard matrix
  fig_core_maps.png   per-category f_c maps over the layer x head grid

Usage:
  python3 distribution_analysis.py \
      --json graphs/consistency/meta-llama_Meta-Llama-3-70B_dead_heads.json \
      --tau 0.15 --core-theta 0.8 --out dist_out
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── activation mask + basic set ops ──────────────────────────────────────────
def active_mask(score: np.ndarray, tau: float) -> np.ndarray:
    """score [L,H] → bool [L,H]; active iff score >= tau * layer-max."""
    mx = score.max(axis=1, keepdims=True)
    return (score >= tau * mx) & (mx > 0)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman via ranks + Pearson; no scipy dependency."""
    if x.std() == 0 or y.std() == 0:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ── per-category summaries ───────────────────────────────────────────────────
def within_jaccard(masks) -> float:
    n = len(masks)
    if n < 2:
        return np.nan
    vals = [jaccard(masks[i], masks[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


def between_jaccard(masks_a, masks_b) -> float:
    vals = [jaccard(a, b) for a in masks_a for b in masks_b]
    return float(np.mean(vals)) if vals else np.nan


def rank_consistency(scores) -> float:
    """Per-layer head-ranking agreement across prompts (threshold-free)."""
    n, L, H = len(scores), scores[0].shape[0], scores[0].shape[1]
    if n < 2:
        return np.nan
    per_pair = []
    for i in range(n):
        for j in range(i + 1, n):
            rhos = [_spearman(scores[i][l], scores[j][l]) for l in range(L)]
            rhos = [r for r in rhos if not np.isnan(r)]
            if rhos:
                per_pair.append(np.mean(rhos))
    return float(np.mean(per_pair)) if per_pair else np.nan


def jaccard_null_z(masks, n_perm=200, seed=0) -> float:
    """z-score of observed within-Jaccard vs. head-label-shuffled null
    (shuffle heads within each layer independently per prompt)."""
    rng = np.random.default_rng(seed)
    obs = within_jaccard(masks)
    L, H = masks[0].shape
    null = []
    for _ in range(n_perm):
        shuffled = []
        for m in masks:
            s = m.copy()
            for l in range(L):
                s[l] = m[l][rng.permutation(H)]
            shuffled.append(s)
        null.append(within_jaccard(shuffled))
    null = np.array(null)
    return float((obs - null.mean()) / (null.std() + 1e-9))


def eta_squared(scores_by_cat) -> tuple:
    """Per-head eta^2 (between-category / total variance) over all prompts.
    Returns (mean_eta2, eta2_map [L,H])."""
    cats = list(scores_by_cat)
    L, H = scores_by_cat[cats[0]][0].shape
    all_scores = np.concatenate([np.stack(scores_by_cat[c]) for c in cats], axis=0)  # [P,L,H]
    grand = all_scores.mean(axis=0)                                                  # [L,H]
    ss_total = ((all_scores - grand) ** 2).sum(axis=0)                               # [L,H]
    ss_between = np.zeros((L, H))
    for c in cats:
        arr = np.stack(scores_by_cat[c])
        ss_between += arr.shape[0] * (arr.mean(axis=0) - grand) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        eta2 = np.where(ss_total > 0, ss_between / ss_total, 0.0)
    return float(np.nanmean(eta2)), eta2


# ── figures ──────────────────────────────────────────────────────────────────
def make_figures(f_maps, jmat, cats, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) trimodal f_c histograms
    ncat = len(cats)
    fig, axs = plt.subplots(1, ncat, figsize=(2.6 * ncat, 2.6), sharey=True)
    if ncat == 1:
        axs = [axs]
    for ax, c in zip(axs, cats):
        ax.hist(f_maps[c].ravel(), bins=np.linspace(0, 1, 21),
                color="#6a51a3", edgecolor="white")
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("activation freq $f_c$")
    axs[0].set_ylabel("# heads")
    fig.suptitle("Distribution of per-head activation frequency (core = spike near 1)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_freq_hist.png", dpi=140); plt.close(fig)

    # 2) within/between Jaccard matrix
    fig, ax = plt.subplots(figsize=(0.9 * ncat + 1.5, 0.9 * ncat + 1.2))
    im = ax.imshow(jmat, cmap="viridis", vmin=0)
    for i in range(ncat):
        for j in range(ncat):
            ax.text(j, i, f"{jmat[i,j]:.2f}", ha="center", va="center",
                    color="white" if jmat[i, j] < jmat.max() * 0.6 else "black",
                    fontsize=8)
    ax.set_xticks(range(ncat)); ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(ncat)); ax.set_yticklabels(cats, fontsize=8)
    ax.set_title("Jaccard overlap of active-head sets\n(diagonal = within-category)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_jaccard.png", dpi=140); plt.close(fig)

    # 3) per-category f_c maps (layer x head)
    fig, axs = plt.subplots(1, ncat, figsize=(2.4 * ncat, 3.4), sharey=True)
    if ncat == 1:
        axs = [axs]
    for ax, c in zip(axs, cats):
        im = ax.imshow(f_maps[c], aspect="auto", cmap="magma", vmin=0, vmax=1,
                       origin="lower")
        ax.set_title(c, fontsize=10); ax.set_xlabel("head")
    axs[0].set_ylabel("layer")
    fig.suptitle("Activation-frequency map $f_c(\\ell,h)$  (bright = core head)", fontsize=11)
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    fig.savefig(f"{out_dir}/fig_core_maps.png", dpi=140, bbox_inches="tight"); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Head-set consistency / distribution study.")
    ap.add_argument("--json", required=True, help="consistency-suite dead_heads JSON")
    ap.add_argument("--tau", type=float, default=0.15, help="active threshold (x layer-max)")
    ap.add_argument("--core-theta", type=float, default=0.8,
                    help="core = heads active in >= theta of category prompts")
    ap.add_argument("--out", default="dist_out")
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(args.json))
    NL, NH = d["n_layers"], d["n_heads"]
    key = "per_head_score" if "per_head_score" in d["tasks"][0] else "per_head_norm"

    # group scores + masks by category
    scores_by_cat = defaultdict(list)
    masks_by_cat = defaultdict(list)
    for t in d["tasks"]:
        cat = t["label"].split("#")[0]
        s = np.asarray(t[key], dtype=float)
        scores_by_cat[cat].append(s)
        masks_by_cat[cat].append(active_mask(s, args.tau))
    cats = list(scores_by_cat)
    print(f"model={d['model']}  {NL}L x {NH}H  metric={d['metric']}  tau={args.tau}")
    print(f"categories: " + ", ".join(f"{c}(n={len(masks_by_cat[c])})" for c in cats))

    # per-category f_c maps + summaries
    f_maps = {c: np.mean(masks_by_cat[c], axis=0) for c in cats}
    report = {"tau": args.tau, "core_theta": args.core_theta, "categories": {}}
    mean_eta2, eta2_map = eta_squared(scores_by_cat)

    print(f"\n{'category':12s}{'within-J':>9s}{'rankConsist':>12s}{'J z-score':>11s}"
          f"{'coreSize':>9s}{'coreDepth':>10s}{'coreH/lyr':>10s}")
    for c in cats:
        wj = within_jaccard(masks_by_cat[c])
        rc = rank_consistency(scores_by_cat[c])
        z = jaccard_null_z(masks_by_cat[c], n_perm=args.n_perm)
        core = f_maps[c] >= args.core_theta                 # [L,H]
        core_size = int(core.sum())
        core_layers = np.where(core.any(axis=1))[0]
        depth_span = f"{core_layers.min()}-{core_layers.max()}" if core_size else "-"
        heads_per_layer = float(core.sum(axis=1)[core.any(axis=1)].mean()) if core_size else 0.0
        report["categories"][c] = dict(
            within_jaccard=wj, rank_consistency=rc, jaccard_z=z,
            core_size=core_size, core_depth_span=depth_span,
            core_heads_per_layer=heads_per_layer)
        print(f"{c:12s}{wj:9.3f}{rc:12.3f}{z:11.1f}{core_size:9d}{depth_span:>10s}"
              f"{heads_per_layer:10.1f}")

    # within vs between Jaccard matrix
    jmat = np.zeros((len(cats), len(cats)))
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            jmat[i, j] = (within_jaccard(masks_by_cat[ci]) if i == j
                          else between_jaccard(masks_by_cat[ci], masks_by_cat[cj]))
    within_mean = np.mean([jmat[i, i] for i in range(len(cats))])
    off = jmat[~np.eye(len(cats), dtype=bool)]
    print(f"\nmean within-category Jaccard  = {within_mean:.3f}")
    print(f"mean between-category Jaccard = {off.mean():.3f}   (ratio {within_mean/ (off.mean()+1e-9):.1f}x)")
    print(f"variance explained by category (mean eta^2 over heads) = {mean_eta2:.3f}")
    report["within_mean"] = float(within_mean)
    report["between_mean"] = float(off.mean())
    report["mean_eta2"] = mean_eta2

    make_figures(f_maps, jmat, cats, str(out))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    np.save(out / "eta2_map.npy", eta2_map)
    for c in cats:
        np.save(out / f"fmap_{c}.npy", f_maps[c])
    print(f"\nreport → {out}/report.json   figures → {out}/fig_*.png")


if __name__ == "__main__":
    main()
