#!/usr/bin/env python3
"""
Analysis for the `serial_replicates` suite.

With one prompt per condition the only available inference is an ordering test
over the six values of k.  Crossing k with paraphrase frames makes the frame a
replicate: each frame yields an independent estimate of a layer's dispersion
slope, so the effect can be tested against between-frame variance and reported
with a confidence interval, and the band boundary acquires a sampling
distribution rather than a single value.

    python3 replicate_analysis.py graphs/.../<model>_dead_heads.json
    python3 replicate_analysis.py <json> --band-only
"""
import argparse, json, math, re, sys
from itertools import permutations

import numpy as np


def pr(v):
    s, s2 = v.sum(), (v * v).sum()
    return (s * s / s2) if s2 > 0 else 0.0


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def t_cdf(t, df):
    """Two-sided p for Student-t without scipy (Hill's approximation via betainc)."""
    x = df / (df + t * t)
    # regularised incomplete beta I_x(df/2, 1/2) by continued fraction
    a, b = df / 2.0, 0.5
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    if x <= 0:
        ib = 0.0
    else:
        # series expansion, adequate for our df
        n, term, total = 0, 1.0, 1.0 / a
        while n < 500:
            n += 1
            term *= (n - b) * x / n
            total += term / (a + n)
            if abs(term / (a + n)) < 1e-14:
                break
        ib = np.exp(a * np.log(x) - lbeta) * total
    return float(min(1.0, max(0.0, ib)))


def load(path):
    d = json.load(open(path))
    L, H = d["n_layers"], d["n_heads"]
    rows = []
    for t in d["tasks"]:
        m = re.match(r"D(\d+)F(\d+)$", t["label"])
        if not m:
            continue
        rows.append((int(m.group(1)), int(m.group(2)),
                     np.array(t["per_head_score"], float)))
    if not rows:
        sys.exit("No D{k}F{f} labels found — is this the serial_replicates suite?")
    ks = sorted({r[0] for r in rows})
    fs = sorted({r[1] for r in rows})
    P = np.full((len(ks), len(fs), L), np.nan)
    for k, f, A in rows:
        P[ks.index(k), fs.index(f)] = [pr(A[l]) / H for l in range(L)]
    return d, np.array(ks, float), fs, P          # P[k, frame, layer]


def band_from(slope, pvals, L, sustain=0.15, window=0.10, call=0.50):
    frac = []
    for l in range(L):
        w = [i for i in range(L)
             if abs(i - l) / (L - 1) <= window and pvals[i] < call]
        frac.append(sum(1 for i in w if slope[i] > 0) / len(w) if w else np.nan)
    need, onset = int(sustain * L), None
    for l in range(L):
        if frac[l] > 0.5 and all(frac[j] > 0.5 for j in range(l, min(L, l + need))):
            onset = l
            break
    close = max((l for l in range(L) if frac[l] > 0.5), default=None) if onset is not None else None
    return onset, close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--band-only", action="store_true")
    a = ap.parse_args()
    d, ks, fs, P = load(a.json)
    nk, nf, L = P.shape
    print(f"{d['model']}   {L} layers, {nk} step values, {nf} frames "
          f"({nk*nf} prompts)")
    print(f"load_info: {d.get('load_info', {})}\n")

    # per-frame, per-layer slope: [frame, layer]
    S = np.stack([[ols_slope(ks, P[:, f, l]) for l in range(L)] for f in range(nf)])

    # ── variance decomposition ──────────────────────────────────────────────
    within = float(np.nanmean(np.nanvar(P, axis=0)))     # across k, within frame
    between = float(np.nanmean(np.nanvar(P.mean(axis=0), axis=0)))  # across frames
    print("Variance in PR/H")
    print(f"  across k within a frame : {within:.3e}")
    print(f"  across frames           : {between:.3e}")
    print(f"  frame / step ratio      : {between/within:8.2f}x")
    print("  (a ratio well above 1 means the frame dominates the manipulation,\n"
          "   and a single-frame design cannot separate them)\n")

    if not a.band_only:
        # ── layer-level test against between-frame variance ─────────────────
        mean = S.mean(0)
        se = S.std(0, ddof=1) / np.sqrt(nf)
        tstat = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
        pv = np.array([t_cdf(t, nf - 1) for t in tstat])
        sig = np.where(pv < 0.05)[0]
        print(f"Layers with a slope distinguishable from zero across frames "
              f"(t-test, df={nf-1}): {len(sig)}/{L}")
        pos = [l for l in sig if mean[l] > 0]
        neg = [l for l in sig if mean[l] < 0]
        print(f"  dispersing   {len(pos)}:  " +
              ", ".join(f"L{l}[{l/(L-1):.2f}]" for l in pos[:14]) +
              (" ..." if len(pos) > 14 else ""))
        print(f"  concentrating {len(neg)}:  " +
              ", ".join(f"L{l}[{l/(L-1):.2f}]" for l in neg[:14]) +
              (" ..." if len(neg) > 14 else ""))
        print("\n  95% CI on the mid-band (40-80%) mean slope:")
        lo, hi = int(0.4 * L), int(0.8 * L)
        band_slopes = S[:, lo:hi].mean(1)
        m, s = band_slopes.mean(), band_slopes.std(ddof=1) / np.sqrt(nf)
        print(f"    {1e3*m:+.3f} +/- {1e3*1.96*s:.3f}  (x10^-3 per step)  "
              f"frames positive: {int((band_slopes>0).sum())}/{nf}")

    # ── band boundary as a sampling distribution ────────────────────────────
    print("\nBand boundary per frame (leave-one-frame-out is the honest CI):")
    onsets, closes = [], []
    for f in range(nf):
        sl = S[f]
        # per-layer p within a frame: permutation over the k ordering
        perms = list(permutations(range(nk)))
        pvv = np.empty(L)
        for l in range(L):
            y = P[:, f, l]
            o = abs(ols_slope(ks, y))
            pvv[l] = sum(1 for q in perms
                         if abs(ols_slope(ks, y[list(q)])) >= o - 1e-15) / len(perms)
        on, cl = band_from(sl, pvv, L)
        onsets.append(np.nan if on is None else on / (L - 1))
        closes.append(np.nan if cl is None else cl / (L - 1))
        print(f"  frame {f}: " + (f"{onsets[-1]:.2f} - {closes[-1]:.2f}"
                                  if on is not None else "never opens"))
    ok = ~np.isnan(onsets)
    if ok.sum():
        o = np.array(onsets)[ok]
        print(f"\n  opens in {int(ok.sum())}/{nf} frames; onset median {np.median(o):.2f}, "
              f"range {o.min():.2f}-{o.max():.2f}")
    else:
        print("\n  opens in 0 frames — the band is not robust to wording.")


if __name__ == "__main__":
    main()
