#!/usr/bin/env python3
"""
test_path_dp.py
===============
Formal correctness tests for the path-length DP (Algorithm 1).

Tests verify that:
  (A) Skip connections always contribute shorter paths and are never pruned.
  (B) Compute-only paths produce the correct maximum path length.
  (C) Analytical distributions match closed-form expressions.
  (D) Active-subgraph pruning only removes compute edges, not skip edges.
  (E) Parallel (GPT-J) ternary branches are correct.

Run:
  python test_path_dp.py          # all tests
  python test_path_dp.py -v       # verbose trace
"""

import argparse
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Minimal standalone DP — mirrors path_analyzer._path_count_dp exactly
# so tests can run without loading any model.
# ---------------------------------------------------------------------------

def path_count_dp(
    n_layers: int,
    active_attn: list,
    active_mlp:  list,
    is_parallel:  bool = False,
    is_attn_only: bool = False,
) -> np.ndarray:
    """
    Exact copy of PathAnalyzer._path_count_dp, standalone for testing.

    Skip connections are ALWAYS active regardless of active_attn / active_mlp.
    They are implemented by `new = counts.copy()`, which propagates the
    current distribution with ZERO path-length increment.

    Compute edges add +1 via `new[1:] += counts[:-1]`.
    """
    if is_parallel or is_attn_only:
        max_path_len = n_layers
    else:
        max_path_len = 2 * n_layers

    counts = np.zeros(max_path_len + 1, dtype=np.float64)
    counts[0] = 1.0          # one path of length 0 at the input

    for layer in range(n_layers):
        has_attn = bool(active_attn[layer])
        has_mlp  = bool(active_mlp[layer]) and not is_attn_only

        if is_parallel:
            # --- Ternary branch -------------------------------------------
            # skip    : new = counts.copy()   (length unchanged)
            # attn    : new[1:] += counts[:-1]  (+1)
            # mlp     : new[1:] += counts[:-1]  (+1)
            # Both attn+mlp both shift from the PRE-update counts,
            # so they are independent ternary options, not sequential.
            new = counts.copy()
            if has_attn:
                new[1:] += counts[:-1]
            if has_mlp:
                new[1:] += counts[:-1]
            counts = new

        else:
            # --- Attention binary branch -----------------------------------
            new = counts.copy()                 # skip attn (always)
            if has_attn:
                new[1:] += counts[:-1]          # compute attn (+1)
            counts = new

            # --- MLP binary branch (sequential only) ----------------------
            if not is_attn_only:
                new = counts.copy()             # skip mlp (always)
                if has_mlp:
                    new[1:] += counts[:-1]      # compute mlp (+1)
                counts = new

    return counts


def to_metrics(counts: np.ndarray) -> dict:
    total = counts.sum()
    if total == 0:
        return {"total": 0, "E_L": 0.0, "H": 0.0, "distribution": counts}
    dist = counts / total
    idxs = np.where(dist > 0)[0]
    E_L  = float(np.dot(idxs, dist[idxs]))
    H    = float(-np.sum(dist[idxs] * np.log2(dist[idxs])))
    return {"total": int(total), "E_L": E_L, "H": H, "distribution": dist}


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = "", verbose: bool = False):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    marker = "  [OK]" if condition else "  [!!]"
    line = f"{marker}  {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not condition:
        print(f"        ^^^ ASSERTION FAILED: {name}")


def assert_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# A.  Skip connections always produce paths of SHORTER length
# ---------------------------------------------------------------------------

def test_skip_connections(verbose: bool = False):
    print("\n─── A. Skip connections produce shorter paths ───────────────────")

    # A1: All layers fully pruned → only skip paths → all paths have length 0
    n = 4
    zeros = [False] * n
    c = path_count_dp(n, zeros, zeros)
    m = to_metrics(c)
    check("A1: all-pruned → single path of length 0",
          m["total"] == 1 and assert_close(m["E_L"], 0.0),
          f"total={m['total']}  E[L]={m['E_L']:.4f}")

    # A2: Paths with only skip edges always exist regardless of pruning
    # Even when every compute edge is pruned, counts[0] >= 1
    n = 8
    partial_attn = [True, False, True, False, False, True, False, True]
    partial_mlp  = [False, True, False, True, False, False, True, False]
    c = path_count_dp(n, partial_attn, partial_mlp)
    check("A2: skip path always exists (counts[0] >= 1)",
          c[0] >= 1,
          f"counts[0]={c[0]:.0f}")

    # A3: Minimum path length in any empirical distribution is always 0
    # (the all-skip path is never removed by pruning)
    n = 6
    half_attn = [True] * 3 + [False] * 3
    half_mlp  = [False] * 3 + [True] * 3
    c = path_count_dp(n, half_attn, half_mlp)
    check("A3: minimum achievable path length is always 0",
          c[0] >= 1,
          f"counts[0]={c[0]:.0f}")

    # A4: With only ONE layer active (say layer 2 attn), we get exactly
    # 2 paths: length-0 (skip) and length-1 (use attn at layer 2)
    n = 5
    single_attn = [False, False, True, False, False]
    all_mlp_off = [False] * n
    c = path_count_dp(n, single_attn, all_mlp_off)
    check("A4: single active layer → 2 paths [skip, compute]",
          c[0] == 1 and c[1] == 1 and c[2:].sum() == 0,
          f"counts={c[:5].astype(int).tolist()}")

    # A5: Skip edges are independent of the active_attn mask —
    # if we flip a layer from active to pruned, counts[0] must stay >= 1
    n = 3
    c_before = path_count_dp(n, [True]*n, [True]*n)
    c_after  = path_count_dp(n, [False]*n, [False]*n)
    check("A5: pruning compute edges never removes length-0 skip path",
          c_after[0] >= 1,
          f"counts[0] before={c_before[0]:.0f}, after={c_after[0]:.0f}")


# ---------------------------------------------------------------------------
# B.  Maximum path length is always capped at 2L (seq) or L (parallel)
# ---------------------------------------------------------------------------

def test_max_path_length(verbose: bool = False):
    print("\n─── B. Maximum path length cap ──────────────────────────────────")

    # B1: Sequential, all active: no paths beyond index 2L
    n = 4
    c = path_count_dp(n, [True]*n, [True]*n, is_parallel=False)
    check("B1: sequential all-active, max non-zero index = 2L",
          np.where(c > 0)[0][-1] == 2*n,
          f"last non-zero={np.where(c>0)[0][-1]}  expected={2*n}")

    # B2: Parallel all-active: max non-zero index = L
    n = 6
    c = path_count_dp(n, [True]*n, [True]*n, is_parallel=True)
    check("B2: parallel all-active, max non-zero index = L",
          np.where(c > 0)[0][-1] == n,
          f"last non-zero={np.where(c>0)[0][-1]}  expected={n}")

    # B3: Partial pruning never exceeds 2L
    n = 8
    partial = [i % 2 == 0 for i in range(n)]
    c = path_count_dp(n, partial, partial, is_parallel=False)
    check("B3: partial pruning never exceeds max_path_len",
          np.where(c > 0)[0][-1] <= 2*n,
          f"last non-zero={np.where(c>0)[0][-1]}  max={2*n}")


# ---------------------------------------------------------------------------
# C.  Analytical distributions match closed-form expressions
# ---------------------------------------------------------------------------

def test_analytical_distributions(verbose: bool = False):
    print("\n─── C. Analytical distributions match closed-form ───────────────")

    # C1: Sequential L layers → Binomial(2L, 0.5) × 4^L
    #     Total paths = 4^L, E[L] = L
    for L in [1, 2, 4, 8, 16]:
        c    = path_count_dp(L, [True]*L, [True]*L, is_parallel=False)
        m    = to_metrics(c)
        binom_counts = np.array([_binom_coeff(2*L, k) for k in range(2*L+1)],
                                 dtype=float)
        match = np.allclose(c, binom_counts, rtol=1e-9)
        check(f"C1: sequential L={L:2d}, total=4^L={4**L:10d}, E[L]=L",
              m["total"] == 4**L and assert_close(m["E_L"], L) and match,
              f"total={m['total']}  E[L]={m['E_L']:.4f}")

    # C2: Parallel L layers → total = 3^L, E[L] = 2L/3
    for L in [1, 2, 4, 6, 12]:
        c = path_count_dp(L, [True]*L, [True]*L, is_parallel=True)
        m = to_metrics(c)
        check(f"C2: parallel   L={L:2d}, total=3^L={3**L:10d}, E[L]=2L/3",
              m["total"] == 3**L and assert_close(m["E_L"], 2*L/3),
              f"total={m['total']}  E[L]={m['E_L']:.4f}  expected={2*L/3:.4f}")

    # C3: Attn-only L layers → Binomial(L, 0.5) × 2^L, E[L] = L/2
    for L in [1, 2, 8]:
        c = path_count_dp(L, [True]*L, [False]*L, is_attn_only=True)
        m = to_metrics(c)
        check(f"C3: attn-only  L={L:2d}, total=2^L={2**L:10d}, E[L]=L/2",
              m["total"] == 2**L and assert_close(m["E_L"], L/2),
              f"total={m['total']}  E[L]={m['E_L']:.4f}  expected={L/2:.4f}")


# ---------------------------------------------------------------------------
# D.  Empirical / active-subgraph pruning — only compute edges removed
# ---------------------------------------------------------------------------

def test_empirical_pruning(verbose: bool = False):
    print("\n─── D. Empirical pruning only removes compute edges ─────────────")

    # D1: Pruning all MLPs should give Binomial(L, 0.5) for attention only
    L = 8
    c = path_count_dp(L, [True]*L, [False]*L, is_parallel=False)
    m = to_metrics(c)
    check("D1: all-MLP-pruned → attn-only Binomial(L, 0.5)",
          m["total"] == 2**L and assert_close(m["E_L"], L/2),
          f"total={m['total']}  E[L]={m['E_L']:.4f}")

    # D2: Pruning all attns should give Binomial(L, 0.5) for MLPs
    L = 8
    c = path_count_dp(L, [False]*L, [True]*L, is_parallel=False)
    m = to_metrics(c)
    check("D2: all-attn-pruned → mlp-only Binomial(L, 0.5)",
          m["total"] == 2**L and assert_close(m["E_L"], L/2),
          f"total={m['total']}  E[L]={m['E_L']:.4f}")

    # D3: Pruning k layers completely (both attn+mlp) should reduce
    #     total paths by factor 1 (skip path still exists for each pruned layer)
    L = 4
    c_full   = path_count_dp(L, [True]*L,  [True]*L)
    c_pruned = path_count_dp(L, [True, True, False, True],
                                [True, True, False, True])
    # Pruning layer 2 entirely removes its binary×binary branch:
    # full = 4^4 = 256, pruned = 4^3 × 1 = 64 (layer 2 only has skip path)
    check("D3: pruning layer 2 entirely reduces total paths from 4^L to 4^(L-1)",
          c_pruned.sum() == 4**(L-1) and c_full.sum() == 4**L,
          f"full={int(c_full.sum())}  pruned={int(c_pruned.sum())}  "
          f"expected={4**(L-1)}")

    # D4: E[L] always decreases (or stays equal) as more edges are pruned
    L = 6
    E_full    = to_metrics(path_count_dp(L, [True]*L, [True]*L))["E_L"]
    E_partial = to_metrics(path_count_dp(L, [True,True,False,True,False,True],
                                            [False,True,True,False,True,True]))["E_L"]
    E_minimal = to_metrics(path_count_dp(L, [True,False,False,False,False,False],
                                            [False]*L))["E_L"]
    check("D4: E[L] decreases monotonically with more pruning",
          E_full >= E_partial >= E_minimal,
          f"full={E_full:.3f}  partial={E_partial:.3f}  minimal={E_minimal:.3f}")

    # D5: Pruning a layer in an active subgraph cannot raise E[L] above analytical
    L = 4
    ana_E_L = to_metrics(path_count_dp(L, [True]*L, [True]*L))["E_L"]
    for trial in range(20):
        rng = np.random.default_rng(trial)
        aa = rng.integers(0, 2, L).astype(bool).tolist()
        am = rng.integers(0, 2, L).astype(bool).tolist()
        emp_E_L = to_metrics(path_count_dp(L, aa, am))["E_L"]
        if emp_E_L > ana_E_L + 1e-9:
            check(f"D5: empirical E[L] never exceeds analytical (trial {trial})",
                  False, f"emp={emp_E_L:.4f} > ana={ana_E_L:.4f}")
            return
    check("D5: empirical E[L] never exceeds analytical (20 random masks)",
          True, f"ana={ana_E_L:.4f}")


# ---------------------------------------------------------------------------
# E.  Parallel (ternary) architecture correctness
# ---------------------------------------------------------------------------

def test_parallel(verbose: bool = False):
    print("\n─── E. Parallel (ternary) architecture ──────────────────────────")

    # E1: 1 parallel layer, both active → [1, 2] distribution
    #     3 paths: 1 skip (len 0) + 1 via attn (len 1) + 1 via mlp (len 1)
    c = path_count_dp(1, [True], [True], is_parallel=True)
    check("E1: parallel L=1, both active → counts=[1,2], total=3",
          c[0] == 1 and c[1] == 2 and c.sum() == 3,
          f"counts={c.tolist()}")

    # E2: 1 parallel layer, only attn active → [1, 1] distribution
    #     2 paths: skip (len 0) + via attn (len 1)
    c = path_count_dp(1, [True], [False], is_parallel=True)
    check("E2: parallel L=1, attn-only → counts=[1,1], total=2",
          c[0] == 1 and c[1] == 1 and c.sum() == 2,
          f"counts={c.tolist()}")

    # E3: 1 parallel layer, only skip → [1] distribution, E[L]=0
    c = path_count_dp(1, [False], [False], is_parallel=True)
    check("E3: parallel L=1, all-pruned → counts=[1], E[L]=0",
          c[0] == 1 and c.sum() == 1,
          f"counts={c.tolist()}")

    # E4: Parallel attn+mlp at the SAME layer give INDEPENDENT +1 contributions
    #     (both shift from the PRE-update counts), so the total path count is 3,
    #     not 4.  Were they sequential, both traversed would add +2 and give 4.
    #
    #     L=1 parallel: max_path_len=L=1, array size=2 → indices {0,1} only.
    #     There is no index-2 slot.  Ternary proof: total=3 (not 4=sequential).
    L = 1
    c_both = path_count_dp(L, [True], [True], is_parallel=True)
    # Sequential equivalent for comparison
    c_seq  = path_count_dp(L, [True], [True], is_parallel=False)
    check("E4: parallel both-active → total=3 (ternary), not 4 (sequential)",
          int(c_both.sum()) == 3 and int(c_seq.sum()) == 4,
          f"parallel total={int(c_both.sum())}  sequential total={int(c_seq.sum())}")
    # Additionally: parallel array has size L+1=2 (no length-2 slot exists by construction)
    check("E4b: parallel L=1 array size=2, proving max path len=L=1",
          len(c_both) == L + 1,
          f"array_len={len(c_both)}  expected={L+1}")

    # E5: 2 parallel layers all active → 3^2=9 paths, E[L]=4/3
    L = 2
    c = path_count_dp(L, [True]*L, [True]*L, is_parallel=True)
    m = to_metrics(c)
    check("E5: parallel L=2, total=9=3^2, E[L]=4/3",
          m["total"] == 9 and assert_close(m["E_L"], 4/3),
          f"total={m['total']}  E[L]={m['E_L']:.4f}  expected={4/3:.4f}")


# ---------------------------------------------------------------------------
# F.  Pythia-2.8b specific — expected analytical values at 0.40 threshold
# ---------------------------------------------------------------------------

def test_pythia_2_8b_values(verbose: bool = False):
    print("\n─── F. Pythia-2.8b specific values (32 layers, sequential) ──────")

    L = 32   # Pythia-2.8b has 32 layers

    # F1: Analytical distribution
    c = path_count_dp(L, [True]*L, [True]*L)
    m = to_metrics(c)
    check("F1: Pythia-2.8b analytical total = 4^32",
          m["total"] == 4**L,
          f"total=4^{L}={4**L}")
    check("F2: Pythia-2.8b analytical E[L] = L = 32",
          assert_close(m["E_L"], float(L)),
          f"E[L]={m['E_L']:.4f}  expected=32.0")

    # F3: With top-40% subgraph (approx 13/64 attn, 13/64 mlp active),
    #     E[L] should be substantially below 32 but above 0
    rng = np.random.default_rng(99)
    n_active = int(0.40 * 2 * L)   # 40% of 64 edges ≈ 26 active edges
    mask_flat = np.zeros(2*L, dtype=bool)
    mask_flat[:n_active] = True
    rng.shuffle(mask_flat)
    aa = mask_flat[:L].tolist()
    am = mask_flat[L:].tolist()
    c_emp = path_count_dp(L, aa, am)
    m_emp = to_metrics(c_emp)
    check("F3: 40%-active subgraph E[L] is between 0 and analytical (32)",
          0 < m_emp["E_L"] < m["E_L"],
          f"E[L]_emp={m_emp['E_L']:.4f}  E[L]_ana={m['E_L']:.4f}")

    # F4: Complexity-matching prediction — more active edges → higher E[L]
    E_L_low  = to_metrics(path_count_dp(L, [True]*8  +[False]*24, [False]*L))["E_L"]
    E_L_mid  = to_metrics(path_count_dp(L, [True]*16 +[False]*16, [False]*L))["E_L"]
    E_L_high = to_metrics(path_count_dp(L, [True]*L,               [False]*L))["E_L"]
    check("F4: more active layers → higher E[L] (Complexity-Matching direction)",
          E_L_low < E_L_mid < E_L_high,
          f"8active={E_L_low:.3f}  16active={E_L_mid:.3f}  32active={E_L_high:.3f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binom_coeff(n: int, k: int) -> int:
    from math import comb
    return comb(n, k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print("  PATH DP CORRECTNESS TESTS")
    print("  Key question: do skip (residual) connections produce shorter paths?")
    print("=" * 68)
    print()
    print("  Mechanism: `new = counts.copy()` is the skip edge.")
    print("  It propagates the existing distribution with ZERO length increment.")
    print("  `new[1:] += counts[:-1]` is the compute edge: shifts by +1.")
    print("  Skip edges are NEVER pruned (not subject to active_attn/active_mlp).")

    test_skip_connections(args.verbose)
    test_max_path_length(args.verbose)
    test_analytical_distributions(args.verbose)
    test_empirical_pruning(args.verbose)
    test_parallel(args.verbose)
    test_pythia_2_8b_values(args.verbose)

    print()
    print("=" * 68)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed  |  {FAIL} failed")
    print("=" * 68)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
