#!/usr/bin/env python3
"""
Candidate B probe — B0 (anchor identification) and B1 (head-level anchor attribution).

Candidate B holds that a model implements attention's no-op by parking idle heads
on a content-free anchor token, and that longer chains recruit heads *out* of that
parked population.  It therefore predicts a within-head correspondence: across the
task ladder, heads that gain contribution are heads whose anchor attention falls.

This records, for every (task, layer, head): the per-head contribution norm
||z_h W_O^h|| at the final position, the full attention distribution at that
position, and summaries of it (anchor position, anchor mass, BOS mass, entropy).

Requires eager attention — sdpa/flash return no weights.  Set automatically.

    python3 candidate_b_probe.py --model allenai/OLMo-2-1124-7B --suite serial_depth \
        --out probe/olmo

Analysis:  python3 candidate_b_probe.py --analyse probe/olmo/<model>_probe.json
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np


# ───────────────────────────── capture ──────────────────────────────────────
def run_probe(args):
    os.environ.setdefault("ASD_FORCE_STRATEGY", "B")
    os.environ.setdefault("ASD_ATTN_IMPL", "eager")
    import torch
    import active_subgraph_dot as asd
    from active_subgraph_dot import load_model, TASK_SUITES

    model = load_model(args.model, device=args.device, hf_token=args.hf_token)
    if not hasattr(model, "_model"):
        raise SystemExit(
            "This probe needs the _HFHookedModel wrapper (raw HF model underneath).\n"
            "Got a TransformerLens model instead — set ASD_FORCE_STRATEGY=B.")

    hf, tok = model._model, model._tok
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    d_head = model.cfg.d_head
    layers = model._layers()
    print(f"  n_layers={n_layers} n_heads={n_heads} d_head={d_head}")
    print(f"  load_info: {asd.LAST_LOAD_INFO}")

    tasks, labels, _ = TASK_SUITES[args.suite]()
    store = {}

    def _make_wrapper(orig_fn, module, ll):
        def _wrapper(*a, **kw):
            try:
                x = a[0] if a else next(iter(kw.values()))
                W = module.weight
                if x.dim() == 3 and W.device.type != "meta":
                    b, s, _ = x.shape
                    xh = x.reshape(b, s, n_heads, d_head)[0].float()
                    Wr = W.reshape(W.shape[0], n_heads, d_head).float()
                    contrib = torch.einsum("shd,ohd->sho", xh, Wr)
                    store[ll] = contrib.norm(dim=-1).detach().cpu()   # [s, n_heads]
            except Exception:
                pass
            return orig_fn(*a, **kw)
        return _wrapper

    patched = []
    for l in range(n_layers):
        attn = model._get_attn(layers[l])
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None:
            continue
        attr = "_old_forward" if hasattr(o_proj, "_old_forward") else "forward"
        orig = getattr(o_proj, attr)
        setattr(o_proj, attr, _make_wrapper(orig, o_proj, l))
        patched.append((o_proj, attr, orig))

    out = {"model": args.model, "suite": args.suite,
           "n_layers": n_layers, "n_heads": n_heads,
           "load_info": dict(asd.LAST_LOAD_INFO), "tasks": []}
    try:
        for i, (text, label) in enumerate(zip(tasks, labels)):
            print(f"  [{i+1}/{len(tasks)}] {label!r}")
            store.clear()
            enc = tok(text, return_tensors="pt")
            enc = {k: v.to(hf.device) for k, v in enc.items()}
            with torch.no_grad():
                res = hf(**enc, output_attentions=True)
            atts = res.attentions
            if atts is None or atts[0] is None:
                raise SystemExit(
                    "output_attentions returned None — the model is not using eager "
                    "attention. Re-run with ASD_ATTN_IMPL=eager (default here) and a "
                    "transformers version that honours it.")
            toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
            seq = len(toks)

            contrib = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
            for l, m in store.items():
                contrib[l] = m[-1].numpy()          # final position

            anchor_pos = np.zeros((n_layers, n_heads), dtype=np.int32)
            anchor_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
            bos_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
            self_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
            entropy = np.zeros((n_layers, n_heads), dtype=np.float32)
            for l in range(min(n_layers, len(atts))):
                A = atts[l][0, :, -1, :].float().cpu().numpy()   # [n_heads, seq]
                anchor_pos[l] = A.argmax(axis=1)
                anchor_mass[l] = A.max(axis=1)
                bos_mass[l] = A[:, 0]
                self_mass[l] = A[:, -1]
                p = np.clip(A, 1e-12, None)
                entropy[l] = -(p * np.log(p)).sum(axis=1)

            out["tasks"].append({
                "index": i, "label": label, "text": text,
                "seq_len": seq, "tokens": toks,
                "contribution": contrib.tolist(),
                "anchor_pos": anchor_pos.tolist(),
                "anchor_mass": anchor_mass.tolist(),
                "bos_mass": bos_mass.tolist(),
                "self_mass": self_mass.tolist(),
                "attn_entropy": entropy.tolist(),
            })
    finally:
        for o_proj, attr, orig in patched:
            setattr(o_proj, attr, orig)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_")
    path = out_dir / f"{safe}_probe.json"
    path.write_text(json.dumps(out))
    print(f"\nProbe data → {path}")


# ───────────────────────────── analysis ─────────────────────────────────────
def analyse(path, band=None):
    import re
    from itertools import permutations
    d = json.load(open(path))
    L, H = d["n_layers"], d["n_heads"]
    recs = sorted((int(re.match(r"D(\d+)", t["label"]).group(1)), t)
                  for t in d["tasks"] if re.match(r"D(\d+)", t["label"]))
    if len(recs) < 3:
        raise SystemExit("need a D-ladder (serial_depth) to run B1")
    D = np.array([k for k, _ in recs], float)
    C = np.stack([np.array(t["contribution"], float) for _, t in recs])   # [k,L,H]
    Am = np.stack([np.array(t["anchor_mass"], float) for _, t in recs])
    Bm = np.stack([np.array(t["bos_mass"], float) for _, t in recs])
    Ap = np.stack([np.array(t["anchor_pos"], int) for _, t in recs])

    print(f"\n{d['model']}  ({L} layers x {H} heads, {len(D)} conditions)")
    print(f"load_info: {d.get('load_info')}")

    # ── B0: what is each model anchoring on? ────────────────────────────────
    toks = recs[0][1]["tokens"]
    flat = Ap[0].ravel()
    uniq, cnt = np.unique(flat, return_counts=True)
    order = np.argsort(-cnt)
    print("\nB0  anchor positions at the final query position (k=1):")
    for j in order[:6]:
        pos = int(uniq[j])
        tk = toks[pos] if pos < len(toks) else "?"
        print(f"    pos {pos:>3} {tk!r:<14} {cnt[j]:>5}/{L*H} heads "
              f"({100*cnt[j]/(L*H):.1f}%)  mean mass "
              f"{Am[0].ravel()[flat == pos].mean():.3f}")
    print(f"    BOS (pos 0) share of heads: "
          f"{100*float((flat == 0).mean()):.1f}%   mean BOS mass {Bm[0].mean():.3f}")

    # ── B1: do heads that gain contribution lose anchor attention? ──────────
    def slopes(X):
        Dm = D - D.mean()
        return np.einsum("k,klh->lh", Dm, X - X.mean(0)) / (Dm ** 2).sum()
    sc, sa, sb = slopes(C), slopes(Am), slopes(Bm)

    def corr_flat(a, b, mask=None):
        a, b = a.ravel(), b.ravel()
        if mask is not None:
            a, b = a[mask.ravel()], b[mask.ravel()]
        ok = np.isfinite(a) & np.isfinite(b)
        return (float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())) if ok.sum() > 3 else (np.nan, 0)

    print("\nB1  head-level correspondence   (Candidate B predicts NEGATIVE)")
    r, n = corr_flat(sc, sa); print(f"    all heads   r(dContribution, dAnchorMass) = {r:+.3f}   n={n}")
    r, n = corr_flat(sc, sb); print(f"    all heads   r(dContribution, dBOSMass)    = {r:+.3f}   n={n}")

    if band:
        lo, hi = band
        inb = np.zeros((L, H), bool); inb[int(lo * (L - 1)):int(hi * (L - 1)) + 1] = True
        r1, n1 = corr_flat(sc, sa, inb)
        r2, n2 = corr_flat(sc, sa, ~inb)
        print(f"\n    inside band {lo:.2f}-{hi:.2f}:  r = {r1:+.3f}  (n={n1})")
        print(f"    outside band:              r = {r2:+.3f}  (n={n2})")
        print("    Candidate B predicts clearly negative inside, ~0 outside.")

    print("\n    by depth quintile:")
    for q in range(5):
        m = np.zeros((L, H), bool); m[int(q * L / 5):int((q + 1) * L / 5)] = True
        r, n = corr_flat(sc, sa, m)
        print(f"      {20*q:>3}-{20*q+20:<3}%  r = {r:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--suite", default="serial_depth")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--hf_token", default=None)
    ap.add_argument("--out", default="probe")
    ap.add_argument("--analyse", default=None, help="path to a *_probe.json")
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    help="band as two relative depths, e.g. --band 0.30 0.94")
    a = ap.parse_args()
    if a.analyse:
        analyse(a.analyse, band=a.band)
    else:
        if not a.model:
            ap.error("--model required unless --analyse is given")
        run_probe(a)
