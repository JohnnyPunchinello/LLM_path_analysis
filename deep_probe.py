#!/usr/bin/env python3
"""
Single-pass probe capturing everything the current analyses are blocked on.

Every quantity in the existing results is a function of the per-head contribution
vector at the final position, which is why four separate questions are all
unanswerable.  One forward pass with hidden states and attention retained yields
all four:

  contribution  [L,H]      the existing measurement, for continuity
  attention     [L,H]      anchor position/mass, BOS mass, entropy  -> Candidate B
  hidden_sim    [L+1,L+1]  layer-layer representational similarity  -> Candidate A
  prediction               argmax token and correctness             -> the confounds

Candidate A's level mismatch needs hidden-state similarity, not head-allocation
similarity: the two move in opposite directions (r=-0.90 over the OLMo ladder),
so head scores cannot proxy for it.  Candidate B needs attention weights, which
requires eager attention -- fused kernels return none.  The post-training result
needs correctness, since instruct checkpoints may simply solve the task more
often.  The similarity matrix is computed inside the probe and only the L x L
result is stored, so output stays small.

    python3 deep_probe.py --model allenai/OLMo-2-1124-7B --suite serial_depth \
        --out probe/olmo
    python3 deep_probe.py --analyse probe/olmo/<model>_deep.json --band 0.48 0.58
"""
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np


# ────────────────────────────── capture ─────────────────────────────────────
def run(args):
    os.environ.setdefault("ASD_FORCE_STRATEGY", "B")
    os.environ.setdefault("ASD_FORCE_PRECISION", "bf16")
    os.environ.setdefault("ASD_ATTN_IMPL", "eager")
    import torch
    import active_subgraph_dot as asd
    from active_subgraph_dot import load_model, TASK_SUITES

    model = load_model(args.model, device=args.device, hf_token=args.hf_token)
    if not hasattr(model, "_model"):
        raise SystemExit("needs the _HFHookedModel wrapper; set ASD_FORCE_STRATEGY=B")
    hf, tok = model._model, model._tok
    L, H, dh = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    layers = model._layers()
    print(f"  n_layers={L} n_heads={H}  load_info={asd.LAST_LOAD_INFO}")

    suite = TASK_SUITES[args.suite]
    if isinstance(suite, dict):
        tasks, labels = suite["tasks"], suite["labels"]
        golds = suite.get("_answers", [None] * len(tasks))
    else:
        tasks, labels, golds = suite()

    store = {}
    def wrap(orig, module, ll):
        def f(*a, **kw):
            try:
                x = a[0] if a else next(iter(kw.values()))
                W = module.weight
                if x.dim() == 3 and W.device.type != "meta":
                    b, s, _ = x.shape
                    xh = x.reshape(b, s, H, dh)[0].float()
                    Wr = W.reshape(W.shape[0], H, dh).float()
                    store[ll] = torch.einsum("shd,ohd->sho", xh, Wr).norm(dim=-1).detach().cpu()
            except Exception:
                pass
            return orig(*a, **kw)
        return f

    patched = []
    for l in range(L):
        o = getattr(model._get_attn(layers[l]), "o_proj", None)
        if o is None:
            continue
        at = "_old_forward" if hasattr(o, "_old_forward") else "forward"
        orig = getattr(o, at)
        setattr(o, at, wrap(orig, o, l)); patched.append((o, at, orig))

    out = {"model": args.model, "suite": args.suite, "n_layers": L, "n_heads": H,
           "load_info": dict(asd.LAST_LOAD_INFO), "tasks": []}
    try:
        for i, (text, label) in enumerate(zip(tasks, labels)):
            print(f"  [{i+1}/{len(tasks)}] {label}")
            store.clear()
            enc = {k: v.to(hf.device) for k, v in tok(text, return_tensors="pt").items()}
            with torch.no_grad():
                res = hf(**enc, output_attentions=True, output_hidden_states=True)
            if res.attentions is None or res.attentions[0] is None:
                raise SystemExit("no attention weights returned; eager attention is required")

            toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
            gold = None if golds[i] is None else str(golds[i]).strip()

            # These tokenizers split " 1" into a bare space token plus the digit,
            # so the argmax at the final position is whitespace and a one-token
            # read records nothing.  Generate greedily instead and judge the
            # first non-space character of the continuation.
            cur = enc["input_ids"]
            gen_ids = []
            for _ in range(4):
                with torch.no_grad():
                    o = hf(input_ids=cur)
                nx = int(o.logits[0, -1].argmax())
                gen_ids.append(nx)
                cur = torch.cat([cur, torch.tensor([[nx]], device=cur.device)], 1)
            gen_text = tok.decode(gen_ids)
            stripped = gen_text.strip()
            pred = stripped[:1] if stripped else ""
            probs = torch.softmax(res.logits[0, -1].float(), -1)
            tv, ti = probs.topk(5)
            top5 = [[tok.decode([int(j)]), round(float(v), 4)] for v, j in zip(tv, ti)]

            contrib = np.full((L, H), np.nan, np.float32)
            for l, m in store.items():
                contrib[l] = m[-1].numpy()

            # Store the FULL attention row at the final query position.  Peak
            # attention is not the right quantity for Candidate B: the peak can
            # move between conditions, so a change in peak mass conflates "this
            # head left its anchor" with "this head's anchor moved".  With the
            # full row, attention to any FIXED position is computable post hoc.
            seq = len(toks)
            AT = np.zeros((L, H, seq), np.float16)
            ap = np.zeros((L, H), np.int32); am = np.zeros((L, H), np.float32)
            en = np.zeros((L, H), np.float32)
            for l in range(min(L, len(res.attentions))):
                A = res.attentions[l][0, :, -1, :].float().cpu().numpy()
                AT[l] = A.astype(np.float16)
                ap[l] = A.argmax(1); am[l] = A.max(1)
                p = np.clip(A, 1e-12, None); en[l] = -(p * np.log(p)).sum(1)

            # layer-layer representational similarity, averaged over positions
            Hs = torch.stack([h[0].float() for h in res.hidden_states])   # [L+1, seq, d]
            Hn = Hs / Hs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sim_mean = torch.einsum("isd,jsd->ijs", Hn, Hn).mean(-1).cpu().numpy()
            # also at the final position alone, which is where every other
            # quantity in this study is measured
            Hl = Hn[:, -1, :]
            sim_last = (Hl @ Hl.T).cpu().numpy()

            out["tasks"].append({
                "index": i, "label": label, "text": text, "tokens": toks,
                "gold": gold, "pred": pred, "gen_text": gen_text, "top5": top5,
                "correct": (gold is not None and pred == gold),
                "contribution": contrib.tolist(),
                "anchor_pos": ap.tolist(), "anchor_mass": am.tolist(),
                "attn_entropy": en.tolist(),
                "attn_last": AT.astype(np.float32).round(5).tolist(),
                "hidden_sim": sim_mean.tolist(),
                "hidden_sim_last": sim_last.tolist(),
            })
            del res
            torch.cuda.empty_cache()
    finally:
        for o, at, orig in patched:
            setattr(o, at, orig)

    p = Path(args.out); p.mkdir(parents=True, exist_ok=True)
    f = p / f"{args.model.replace('/','_')}_deep.json"
    f.write_text(json.dumps(out))
    nc = sum(t["correct"] for t in out["tasks"])
    print(f"\n  accuracy {nc}/{len(out['tasks'])}")
    print(f"  -> {f}  ({f.stat().st_size/1e6:.1f} MB)")


# ───────────────────────────── analysis ─────────────────────────────────────
def analyse(path, band):
    rng = np.random.default_rng(0)
    d = json.load(open(path))
    L, H = d["n_layers"], d["n_heads"]
    recs = sorted((int(re.match(r"D(\d+)", t["label"]).group(1)), t)
                  for t in d["tasks"] if re.match(r"D(\d+)", t["label"]))
    if len(recs) < 3:
        print("not a D-ladder; accuracy only")
        for t in d["tasks"]:
            print(f"    {t['label']:<12} gold {t['gold']!r:<5} gen {t['gen_text']!r}")
        return
    K = np.array([k for k, _ in recs], float); dm = K - K.mean()
    T = [t for _, t in recs]
    print(f"{d['model']}   L={L} H={H}   load_info={d.get('load_info')}")

    print("\n[0] ACCURACY")
    for k, t in recs:
        mark = "correct" if t["correct"] else "wrong"
        print(f"    k={k}  gold={t['gold']!r:<4} gen={t['gen_text']!r:<14} {mark}"
              f"   top5={[x[0] for x in t['top5']]}")
    print(f"    total {sum(t['correct'] for t in T)}/{len(T)}")

    def slope_of(key):
        X = np.stack([np.array(t[key], float) for t in T])
        return np.einsum("k,k...->...", dm, X - X.mean(0)) / (dm ** 2).sum()
    sc = slope_of("contribution")

    def cf(a, b, m=None):
        a, b = a.ravel(), b.ravel()
        if m is not None: a, b = a[m.ravel()], b[m.ravel()]
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 else np.nan

    print("\n[1] CANDIDATE B  attention to a FIXED anchor")
    AT = np.stack([np.array(t["attn_last"], float) for t in T])      # [k,L,H,seq]
    anchor = AT[0].argmax(-1)                                        # per head, from k=1
    toks = T[0]["tokens"]
    u, c = np.unique(anchor.ravel(), return_counts=True)
    for jx in np.argsort(-c)[:3]:
        p = int(u[jx])
        print(f"    anchor pos {p:>3} {toks[p] if p < len(toks) else '?':<12} "
              f"{c[jx]:>5}/{L*H} heads ({100*c[jx]/(L*H):.1f}%)")
    fixed = np.take_along_axis(AT, anchor[None, :, :, None], axis=3)[..., 0]
    sf = np.einsum("k,klh->lh", dm, fixed - fixed.mean(0)) / (dm ** 2).sum()
    se = slope_of("attn_entropy")
    print(f"    r(dContribution, dAttentionToFixedAnchor) = {cf(sc, sf):+.3f}  "
          "(B predicts negative)")
    print(f"    r(dContribution, dEntropy)                = {cf(sc, se):+.3f}  "
          "(B predicts positive)")
    if band:
        lo, hi = band
        m = np.zeros((L, H), bool); m[int(lo*(L-1)):int(hi*(L-1))+1] = True
        print(f"      inside band {lo:.2f}-{hi:.2f}: {cf(sc, sf, m):+.3f}"
              f"     outside: {cf(sc, sf, ~m):+.3f}")

    print("\n[2] CANDIDATE A  hidden-state locality")
    for key, nm in (("hidden_sim", "position-averaged"), ("hidden_sim_last", "final position")):
        if key not in T[0]: continue
        S = np.mean([np.array(t[key], float) for t in T], axis=0)
        n = S.shape[0]; iu = np.triu_indices(n, 1)
        dist = np.abs(iu[0] - iu[1]).astype(float)
        rho = float(np.corrcoef(np.argsort(np.argsort(dist)).astype(float),
                                np.argsort(np.argsort(S[iu])).astype(float))[0, 1])
        print(f"    {nm:<18} rho = {rho:+.3f}   RLS = {-np.log2(max(1e-12, S[iu].mean())):.3f}")
        if not band: continue
        lo, hi = band; a, b = int(lo*(n-1)), int(hi*(n-1)); span = b - a + 1
        def excess(mask):
            din, dout = {}, {}
            for x in range(n):
                for y in range(x+1, n):
                    k = y - x
                    if mask[x] and mask[y]: din.setdefault(k, []).append(S[x, y])
                    elif not mask[x] and not mask[y]: dout.setdefault(k, []).append(S[x, y])
            com = [k for k in din if k in dout and len(din[k]) >= 3 and len(dout[k]) >= 3]
            if not com: return np.nan
            return (np.mean([np.mean(din[k]) for k in com])
                    - np.mean([np.mean(dout[k]) for k in com]))
        m = np.zeros(n, bool); m[a:b+1] = True
        obs = excess(m)
        # Two nulls.  The unrestricted one is conservative: a random block of the
        # same size usually overlaps the observed band, so it inherits part of the
        # effect.  The disjoint one asks whether the band beats blocks that share
        # no layers with it, which is the cleaner comparison when the band is wide.
        nl, nd = [], []
        for st in range(0, n - span + 1):
            m2 = np.zeros(n, bool); m2[st:st+span] = True
            v = excess(m2)
            if not np.isfinite(v): continue
            nl.append(v)
            if not (m2 & m).any(): nd.append(v)
        p  = sum(1 for v in nl if v >= obs) / len(nl) if nl else np.nan
        pd = sum(1 for v in nd if v >= obs) / len(nd) if nd else np.nan
        print(f"    {'':<18} in-band excess {obs:+.4f}   all blocks "
              f"{np.mean(nl):+.4f} p={p:.3f} (n={len(nl)})")
        if nd:
            print(f"    {'':<18} {'':<14}   disjoint blocks "
                  f"{np.mean(nd):+.4f} p={pd:.3f} (n={len(nd)})")
        else:
            print(f"    {'':<18} {'':<14}   no disjoint block of this size exists "
                  f"(band covers {span}/{n} layers)")
    print("      p is the share of equal-size random contiguous blocks doing as well;")
    print("      a band that is merely late and contiguous will not be significant.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--suite", default="serial_depth")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--hf_token", default=None)
    ap.add_argument("--out", default="probe")
    ap.add_argument("--analyse"); ap.add_argument("--band", nargs=2, type=float)
    a = ap.parse_args()
    if a.analyse: analyse(a.analyse, a.band)
    elif a.model: run(a)
    else: ap.error("--model or --analyse required")
