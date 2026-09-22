#!/usr/bin/env python3
"""
Per-item probe: per-head contribution at the final position plus the model's
answer, so activation patterns can be studied on the items the model solves.

  contribution  [L,H]      per-head output norm at the measured position: the
                           one whose next token is the answer digit (the
                           prompt's last token, or a trailing space token for
                           tokenizers that split " 6"; see align_answer)
  prediction               greedy continuation, digit-constrained argmax,
                           gold rank/probability, and correctness under each

Use a many-items-per-depth suite (serial_depth_rep, serial_depth_fs): with one
prompt per depth, conditioning on correctness leaves too few points to fit.

    python3 deep_probe.py --model allenai/OLMo-2-1124-7B --suite serial_depth_fs \
        --out probe/olmo
    python3 deep_probe.py --analyse probe/olmo/<model>_deep.json [more.json ...] \
        --plot probe/olmo/correct_only.png
"""
import argparse, json, math, os, re, sys
from pathlib import Path
import numpy as np


# ────────────────────────── answer alignment ────────────────────────────────
def align_answer(tok, text, digits="0123456789"):
    """Make the answer digit the NEXT token after the measured position.

    Standard practice (multi-hop / binding / entity-tracking circuit work) is to
    measure at the position whose next-token prediction is the answer.  How
    " 6" tokenizes decides where that is.  GPT-2 and Pythia have a single " 6"
    token, so it is the prompt's last token.  Most current tokenizers split it
    into a bare space plus "6" (Llama-3 and Qwen "Ġ"+"6"; Mistral, Yi, Phi-3,
    Gemma "▁"+"6"); the space token is then fed as part of the input and the
    measurement moves onto it.  Without this, the Sep 17 runs measured at a
    position whose top prediction was the space (72-98% probability) in all
    eight models.  Derived from the tokenizer, per prompt, not assumed.

    Returns (feed_ids, {digit: answer_token_id}, appended_token_strings).
    Raises if the digit is not a single token after a shared feed, rather than
    measuring somewhere that does not predict the answer.
    """
    base = tok(text)["input_ids"]
    feed, ans = None, {}
    for d in digits:
        full = tok(text + " " + d)["input_ids"]
        f, a = full[:-1], full[-1]
        if full[:len(base)] != base or tok.decode([a]).strip() != d:
            raise SystemExit(f"answer {d!r} is not a single token after the prompt "
                             f"(tokens {tok.convert_ids_to_tokens(full[len(base)-1:])}); "
                             f"cannot align the measured position with the answer")
        if feed is None:
            feed = f
        elif f != feed:
            raise SystemExit(f"digits need different inputs before the answer "
                             f"({tok.convert_ids_to_tokens(feed[len(base):])} vs "
                             f"{tok.convert_ids_to_tokens(f[len(base):])})")
        ans[d] = a
    extra = feed[len(base):]
    if extra and tok.decode(extra).strip():
        raise SystemExit(f"alignment would append non-whitespace tokens "
                         f"{tok.convert_ids_to_tokens(extra)}")
    return feed, ans, tok.convert_ids_to_tokens(extra)


# ────────────────────────────── capture ─────────────────────────────────────
def run(args):
    os.environ.setdefault("ASD_FORCE_STRATEGY", "B")
    os.environ.setdefault("ASD_FORCE_PRECISION", "bf16")
    # eager by default for continuity with earlier runs: the attention kernel
    # changes the contribution metric materially, so it is recorded, not varied.
    os.environ.setdefault("ASD_ATTN_IMPL", args.attn_impl)
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

    # The measured position is the one whose next token is the answer digit;
    # see align_answer.  Report once which form this tokenizer needs.
    _f, _a, _x = align_answer(tok, tasks[0])
    print(f"  measured position: {'prompt + ' + str(_x) if _x else 'last prompt token'}"
          f"   answer token form: {tok.convert_ids_to_tokens([_a['6']])[0]!r}")

    # ── output projection: name and weight layout differ by architecture ──────
    # o_proj (Llama/Mistral/Qwen/OLMo/Gemma/Phi/Yi/Mixtral) is nn.Linear, weight
    # [out, in].  GPT-2's c_proj is a Conv1D with weight [in, out] -- using the
    # Linear layout there gives ~87% relative error, silently.  Pythia/NeoX
    # `dense` is a plain Linear.  Rather than trust a shape heuristic (the matrix
    # is square whenever d_model == H*d_head, so shape cannot disambiguate), we
    # verify functionally: reconstruct the module output from the per-head
    # decomposition and keep the layout that reproduces it.  If neither does, the
    # per-head split is invalid for this architecture and we abort instead of
    # emitting numbers.
    PROJ_NAMES = ("o_proj", "dense", "c_proj", "out_proj", "wo")
    def find_proj(attn):
        for n in PROJ_NAMES:
            p = getattr(attn, n, None)
            if p is not None and hasattr(p, "weight"):
                return p, n
        return None, None

    store = {}
    layout = {"mode": None, "err": None}     # resolved on the first real forward

    def per_head(x, W, mode):
        Wm = W if mode == "linear" else W.t()
        Wr = Wm.reshape(Wm.shape[0], H, dh).float()
        xh = x.reshape(x.shape[0], x.shape[1], H, dh)[0].float()
        return torch.einsum("shd,ohd->sho", xh, Wr)      # [seq, heads, d_model]

    def wrap(orig, module, ll):
        def f(*a, **kw):
            x = a[0] if a else next(iter(kw.values()))
            out = orig(*a, **kw)
            try:
                W = module.weight
                if x.dim() == 3 and W.device.type != "meta":
                    if layout["mode"] is None:
                        ref = out[0].float()
                        bias = getattr(module, "bias", None)
                        best, berr = None, float("inf")
                        for mode in ("linear", "conv1d"):
                            try:
                                rec = per_head(x, W, mode).sum(1)
                                if bias is not None:
                                    rec = rec + bias.float()
                                e = float((rec - ref).abs().max() / ref.abs().max().clamp_min(1e-9))
                            except Exception:
                                e = float("inf")
                            if e < berr:
                                best, berr = mode, e
                        if berr > 1e-2:
                            raise RuntimeError(
                                f"per-head decomposition does not reproduce the output of "
                                f"{type(module).__name__} (best relative error {berr:.2e} "
                                f"under both weight layouts). The attention output of this "
                                f"architecture does not decompose into independent per-head "
                                f"slices -- latent/compressed attention (e.g. DeepSeek MLA) "
                                f"behaves this way. Refusing to emit contribution scores.")
                        layout["mode"], layout["err"] = best, berr
                        print(f"  per-head layout verified: {best} "
                              f"(reconstruction error {berr:.2e})")
                    store[ll] = per_head(x, W, layout["mode"]).norm(dim=-1).detach().cpu()
            except RuntimeError:
                raise
            except Exception:
                pass
            return out
        return f

    patched, pname = [], None
    for l in range(L):
        o, n = find_proj(model._get_attn(layers[l]))
        if o is None:
            continue
        pname = pname or n
        at = "_old_forward" if hasattr(o, "_old_forward") else "forward"
        orig = getattr(o, at)
        setattr(o, at, wrap(orig, o, l)); patched.append((o, at, orig))
    if not patched:
        raise SystemExit("no output projection found in any layer; unsupported architecture")
    print(f"  output projection: {pname}  ({len(patched)}/{L} layers hooked)")

    out = {"model": args.model, "suite": args.suite, "n_layers": L, "n_heads": H,
           "load_info": dict(asd.LAST_LOAD_INFO), "proj_name": pname,
           "head_layout": layout, "tasks": []}
    try:
        for i, (text, label) in enumerate(zip(tasks, labels)):
            print(f"  [{i+1}/{len(tasks)}] {label}")
            store.clear()
            feed, ANS, appended = align_answer(tok, text)
            ids = torch.tensor([feed], device=hf.device)
            enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            with torch.no_grad():
                res = hf(**enc, use_cache=True)

            # SNAPSHOT NOW.  The greedy-generation loop below calls the model
            # another 16 times, and the o_proj hooks fire on every one of those
            # calls, overwriting `store`.  Reading `store` after generation would
            # record the contribution at the final position of prompt+15 generated
            # tokens instead of at the prompt's final position -- a different
            # sequence of a different length.  This silently corrupted an earlier
            # run: its per-rule dispersion ordering came out with the opposite sign
            # and ~5x the variance of a run that did not generate.
            snap = {l: m.clone() for l, m in store.items()}

            toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
            gold = None if golds[i] is None else str(golds[i]).strip()

            # The digit-constrained read below is the primary one, taken at the
            # measured position.  Greedy generation is kept as a behavioural
            # check: the model may still emit something else first.
            # 4 tokens truncated verbose models before they answered (Qwen2.5-7B
            # emitted "  . What is" in every case and was scored as a miss), so
            # the budget is 16 and several independent readings are recorded.
            # No single extraction rule is load-bearing: a result that depends on
            # which one is used is a result about the judge, not the model.
            # KV-cached; the first token comes from the probe forward itself.
            # Stops once a digit other than the echoed step count has appeared,
            # which is everything the readings below need.
            _kmatch = re.search(r"Take (\d+) steps", text)
            _k = _kmatch.group(1) if _kmatch else None
            gen_ids, logits, pkv = [], res.logits, res.past_key_values
            for _ in range(16):
                nx = int(logits[0, -1].argmax())
                gen_ids.append(nx)
                if tok.eos_token_id is not None and nx == tok.eos_token_id:
                    break
                if any(c.isdigit() and c != _k for c in tok.decode(gen_ids)):
                    break
                with torch.no_grad():
                    o = hf(input_ids=torch.tensor([[nx]], device=hf.device),
                           past_key_values=pkv, use_cache=True)
                logits, pkv = o.logits, o.past_key_values
            del pkv
            gen_text = tok.decode(gen_ids)
            stripped = gen_text.strip()
            pred = stripped[:1] if stripped else ""            # first character
            _digits = re.findall(r"\d", gen_text)
            pred_first_digit = _digits[0] if _digits else None   # first digit anywhere
            # first digit that is not simply the step count echoed back
            _nonecho = [x for x in _digits if x != _k]
            pred_nonecho = _nonecho[0] if _nonecho else None
            probs = torch.softmax(res.logits[0, -1].float(), -1)
            tv, ti = probs.topk(5)
            top5 = [[tok.decode([int(j)]), round(float(v), 4)] for v, j in zip(tv, ti)]

            # digit-constrained read over the exact answer tokens for this prompt
            pred_con = max(ANS, key=lambda _d: float(probs[ANS[_d]]))
            gold_rank, gold_prob, digit_mass = None, None, float(sum(probs[a] for a in ANS.values()))
            if gold is not None and gold in ANS:
                gold_prob = float(probs[ANS[gold]])
                gold_rank = int((probs > gold_prob).sum()) + 1

            # Whether a BOS token exists at all decides whether attention to
            # position 0 means anything: without one, position 0 is an ordinary
            # content token and "BOS mass" measures something else entirely.
            _b = getattr(tok, "bos_token_id", None)
            has_bos = _b is not None and int(enc["input_ids"][0][0]) == int(_b)

            contrib = np.full((L, H), np.nan, np.float32)
            for l, m in snap.items():          # snapshot, NOT the post-generation store
                contrib[l] = m[-1].numpy()
            _ok = int(np.isfinite(contrib).all(axis=1).sum())
            if _ok < L:
                raise SystemExit(
                    f"\n  Contribution hook fired on only {_ok}/{L} layers.\n"
                    f"  Usual cause: this model sets head_dim={dh} while "
                    f"hidden_size/n_heads={model.cfg.d_model // H}, so the per-head\n"
                    f"  reshape fails.  Run check_compat.py on this repo.  Earlier\n"
                    f"  revisions swallowed this and wrote all-NaN scores.")

            out["tasks"].append({
                "index": i, "label": label, "text": text, "n_tokens": len(toks),
                "gold": gold, "pred": pred, "gen_text": gen_text, "top5": top5,
                "correct": (gold is not None and pred == gold),
                "pred_constrained": pred_con,
                "correct_constrained": (gold is not None and pred_con == gold),
                "pred_first_digit": pred_first_digit,
                "correct_first_digit": (gold is not None and pred_first_digit == gold),
                "pred_nonecho": pred_nonecho,
                "correct_nonecho": (gold is not None and pred_nonecho == gold),
                # right both at the measured position and in what it generates
                "correct_strict": (gold is not None and pred_con == gold
                                   and pred_first_digit == gold),
                "gold_rank": gold_rank, "gold_prob": gold_prob,
                # probability on any digit at the measured position: low values
                # mean the model is not about to answer here
                "digit_mass": digit_mass,
                "measured_token": toks[-1], "appended": appended,
                "has_bos": has_bos, "pos0_token": toks[0] if toks else None,
                "contribution": contrib.tolist(),
            })
            del res
            torch.cuda.empty_cache()
    finally:
        for o, at, orig in patched:
            setattr(o, at, orig)

    p = Path(args.out); p.mkdir(parents=True, exist_ok=True)
    f = p / f"{args.model.replace('/','_')}_deep.json"
    f.write_text(json.dumps(out))
    nt = len(out["tasks"])
    out["head_layout"] = dict(layout)
    nc = sum(t["correct"] for t in out["tasks"])
    ncc = sum(t["correct_constrained"] for t in out["tasks"])
    rk = [t["gold_rank"] for t in out["tasks"] if t["gold_rank"]]
    nfd = sum(t["correct_first_digit"] for t in out["tasks"])
    nne = sum(t["correct_nonecho"] for t in out["tasks"])
    print(f"\n  accuracy   first-char {nc}/{nt}   first-digit {nfd}/{nt}   "
          f"first-non-echo {nne}/{nt}   digit-constrained {ncc}/{nt}   "
          f"strict {sum(t['correct_strict'] for t in out['tasks'])}/{nt}")
    byd = {}
    for t in out["tasks"]:
        m = re.match(r"D(\d+)", t["label"])
        if m: byd.setdefault(int(m.group(1)), []).append(t["correct_strict"])
    if byd:
        print("  strict by D: " + "  ".join(f"D{k} {sum(v)}/{len(v)}" for k, v in sorted(byd.items())))
    if rk: print(f"  gold-token rank: median {sorted(rk)[len(rk)//2]}, worst {max(rk)}")
    print(f"  BOS present: {sum(t['has_bos'] for t in out['tasks'])}/{nt}"
          f"   position 0 is {out['tasks'][0].get('pos0_token')!r}")
    print(f"  -> {f}  ({f.stat().st_size/1e6:.1f} MB)")


# ───────────────────────────── analysis ─────────────────────────────────────
# Correct-only activation analysis.  The per-head contribution is summarised per
# layer two ways, both normalised per prompt:
#   PR     participation ratio / H   how evenly the layer's heads share the work
#   share  layer's share of total contribution mass across all layers
# Only items the model answers correctly enter the depth profile, and only at
# depths where accuracy clears chance: below chance-level, "correct" items are
# mostly lucky guesses and conditioning on them selects noise.

def _binom_sf(x, n, p):
    """P(X >= x) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, j) * p**j * (1 - p)**(n - j) for j in range(x, n + 1))


def _above_chance(D, ok, chance, min_correct=3):
    """Depths whose accuracy beats chance (one-sided binomial p < .05)."""
    return [int(k) for k in np.unique(D)
            if ok[D == k].sum() >= min_correct
            and _binom_sf(int(ok[D == k].sum()), int((D == k).sum()), chance) < .05]


def _load(path, criterion):
    d = json.load(open(path))
    L, H = d["n_layers"], d["n_heads"]
    T = [t for t in d["tasks"] if re.match(r"D(\d+)", t["label"])]
    key = criterion if criterion in T[0] else (
        "correct_constrained" if "correct_constrained" in T[0] else "correct")
    C = np.array([t["contribution"] for t in T], float)            # [n, L, H]
    if not np.isfinite(C).all():
        raise SystemExit(f"{path}: contribution contains NaN -- this run hit the "
                         f"head_dim bug and must be re-run with the fixed loader")
    s, s2 = C.sum(-1), (C * C).sum(-1)
    return dict(
        model=d["model"], suite=d.get("suite"), L=L, H=H, key=key,
        D=np.array([int(re.match(r"D(\d+)", t["label"]).group(1)) for t in T]),
        ok=np.array([bool(t[key]) for t in T]),
        echo=np.array([t.get("pred_first_digit") == re.search(r"Take (\d+) steps", t["text"]).group(1)
                       for t in T]),
        gold_prob=np.array([t.get("gold_prob") or np.nan for t in T], float),
        digit_mass=np.array([t.get("digit_mass", np.nan) for t in T], float),
        PR=np.where(s2 > 0, s * s / np.where(s2 > 0, s2, 1), 0) / H,        # [n, L]
        share=s / s.sum(1, keepdims=True),                                  # [n, L]
        mass=s,                                                             # [n, L]
        chance=1 / len({t["gold"] for t in T}))


def _depth_slope(Y, D, rng, nperm):
    """Item-level OLS slope of each layer on D, with a joint permutation null
    (D labels shuffled across items, same shuffle for every layer)."""
    dc = D - D.mean(); Yc = Y - Y.mean(0); ss = dc @ dc
    obs = dc @ Yc / ss
    P = np.stack([rng.permutation(dc) for _ in range(nperm)])
    null = P @ Yc / ss                                                # [nperm, L]
    p = ((np.abs(null) >= np.abs(obs) - 1e-15).sum(0) + 1) / (nperm + 1)
    nsig = int((p < .05).sum())
    nsig_null = [int(((np.abs(null) >= np.abs(null[i]) - 1e-15).sum(0) / nperm < .05).sum())
                 for i in range(min(nperm, 500))]
    m_null = null.mean(1)
    return dict(slope=obs, p=p, nsig=nsig, n_pos=int(((p < .05) & (obs > 0)).sum()),
                n_neg=int(((p < .05) & (obs < 0)).sum()),
                p_nsig=(sum(v >= nsig for v in nsig_null) + 1) / (len(nsig_null) + 1),
                mean=float(obs.mean()),
                p_mean=float(((np.abs(m_null) >= abs(obs.mean()) - 1e-15).sum() + 1) / (nperm + 1)))


def _correct_vs_wrong(Y, D, ok, rng, nperm, depths):
    """Per-layer (correct - wrong) difference, averaged over the given depths
    that have at least 3 of each; null shuffles correctness within each depth."""
    strata = [np.where(D == k)[0] for k in depths]
    strata = [ix for ix in strata if ok[ix].sum() >= 3 and (~ok[ix]).sum() >= 3]
    if not strata:
        return None
    def diff(o):
        return np.mean([Y[ix][o[ix]].mean(0) - Y[ix][~o[ix]].mean(0) for ix in strata], 0)
    obs = diff(ok); null = []
    for _ in range(nperm):
        o = ok.copy()
        for ix in strata: o[ix] = rng.permutation(ok[ix])
        null.append(diff(o))
    null = np.array(null)
    p = ((np.abs(null) >= np.abs(obs) - 1e-15).sum(0) + 1) / (nperm + 1)
    return dict(diff=obs, p=p, nsig=int((p < .05).sum()),
                depths=[int(D[ix[0]]) for ix in strata],
                mean=float(obs.mean()),
                p_mean=float(((np.abs(null.mean(1)) >= abs(obs.mean()) - 1e-15).sum() + 1) / (nperm + 1)))


def _signmap(r):
    return "".join("+" if (s > 0 and p < .05) else "-" if (s < 0 and p < .05)
                   else "." for s, p in zip(r["slope"], r["p"]))


def analyse_one(path, criterion="correct_strict", nperm=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = _load(path, criterion)
    D, ok, c = x["D"], x["ok"], x["chance"]
    print(f"\n{'#'*78}\n# {x['model']}   suite={x['suite']}   L={x['L']} H={x['H']}"
          f"   correct = {x['key']}   chance = {c:.3f}\n{'#'*78}")

    print("\n[0] ACCURACY BY DEPTH")
    print(f"    {'D':>3}{'n':>5}{'correct':>9}{'acc':>7}{'p>chance':>10}{'genuine':>9}"
          f"{'echo k':>8}{'P(gold)':>9}{'P(digit)':>10}")
    for k in np.unique(D):
        m = D == k; n, nc = int(m.sum()), int(ok[m].sum()); a = nc / n
        pb = _binom_sf(nc, n, c)
        # share of correct items that are real solves rather than guesses
        gen = max(0.0, (a - c) / (1 - c)) / a if a > 0 else 0.0
        print(f"    {k:>3}{n:>5}{nc:>9}{a:>7.2f}{pb:>10.4f}{gen:>9.2f}"
              f"{x['echo'][m].mean():>8.2f}{np.nanmean(x['gold_prob'][m]):>9.3f}"
              f"{np.nanmean(x['digit_mass'][m]) if np.isfinite(x['digit_mass'][m]).any() else np.nan:>10.3f}")
    above = _above_chance(D, ok, c)
    print(f"    total {ok.sum()}/{len(ok)}   depths above chance (p<.05): {above or 'none'}")

    res = dict(model=x["model"], suite=x["suite"], L=x["L"], D=D, ok=ok, chance=c,
               above=above, acc={int(k): float(ok[D == k].mean()) for k in np.unique(D)})
    sel = ok & np.isin(D, above)
    print("\n[1] DEPTH PROFILE ON CORRECT ITEMS  (above-chance depths only)")
    if len(above) < 3 or sel.sum() < 12:
        print(f"    not enough: {len(above)} depth(s) above chance, {sel.sum()} items. "
              "Need >= 3 depths and >= 12 items for a slope.")
    else:
        for met in ("PR", "share"):
            r = _depth_slope(x[met][sel], D[sel].astype(float), rng, nperm)
            ra = _depth_slope(x[met], D.astype(float), rng, nperm)
            agree = float(np.corrcoef(r["slope"], ra["slope"])[0, 1])
            res[met] = dict(correct=r, all=ra, agree=agree)
            # shares sum to 1 per item, so their mean over layers is constant
            mean = (f"mean slope {r['mean']:+.2e} (p={r['p_mean']:.3f})   "
                    if met == "PR" else "")
            print(f"    {met:<6} n={sel.sum()} items at D={above}")
            print(f"           {mean}layers p<.05: {r['nsig']}/{x['L']} "
                  f"(+{r['n_pos']}/-{r['n_neg']}, p={r['p_nsig']:.3f})")
            print(f"           |{_signmap(r)}|")
            print(f"           all items: {ra['nsig']} layers p<.05"
                  + (f", mean slope {ra['mean']:+.2e} (p={ra['p_mean']:.3f})" if met == "PR" else "")
                  + f"   r(correct-only, all) = {agree:+.2f}")

    print("\n[2] CORRECT vs WRONG AT MATCHED DEPTH  (above-chance depths only)")
    for met in ("PR", "share"):
        r = _correct_vs_wrong(x[met], D, ok, rng, nperm, above)
        if r is None:
            print(f"    {met:<6} no above-chance depth has >= 3 correct and >= 3 wrong"); continue
        res.setdefault("cvw", {})[met] = r
        top = np.argsort(r["p"])[:5]
        mean = f"mean diff {r['mean']:+.2e} (p={r['p_mean']:.3f})   " if met == "PR" else ""
        print(f"    {met:<6} depths {r['depths']}   {mean}layers p<.05: {r['nsig']}/{x['L']}")
        print("           strongest: " + ", ".join(
            f"L{l} ({l/(x['L']-1):.2f}) {r['diff'][l]:+.1e}" for l in top))
    return res


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, ax = plt.subplots(n, 3, figsize=(15, 3.2 * n), squeeze=False)
    for i, r in enumerate(results):
        name = f"{r['model'].split('/')[-1]}\n{r['suite']}"
        ks = sorted(r["acc"])
        a = ax[i, 0]
        a.bar(ks, [r["acc"][k] for k in ks],
              color=["C0" if k in r["above"] else "0.75" for k in ks])
        a.axhline(r["chance"], ls="--", c="k", lw=1); a.set_ylim(0, 1)
        a.set_xlabel("D (steps)"); a.set_ylabel("accuracy"); a.set_title(name, fontsize=9)
        for j, (met, lab) in enumerate((("PR", "PR slope vs D"), ("cvw", "correct - wrong (PR)"))):
            a = ax[i, j + 1]
            if met == "PR" and "PR" in r:
                for key, sty in (("correct", "-"), ("all", "--")):
                    s = r["PR"][key]; xs = np.linspace(0, 1, len(s["slope"]))
                    a.plot(xs, s["slope"], sty, label=f"{key} items")
                    sig = s["p"] < .05
                    if key == "correct": a.plot(xs[sig], s["slope"][sig], "o", ms=3, c="C3")
                a.legend(fontsize=7)
            elif met == "cvw" and "cvw" in r and "PR" in r["cvw"]:
                s = r["cvw"]["PR"]; xs = np.linspace(0, 1, len(s["diff"]))
                a.plot(xs, s["diff"]); sig = s["p"] < .05
                a.plot(xs[sig], s["diff"][sig], "o", ms=3, c="C3")
            else:
                a.text(.5, .5, "not enough correct items", ha="center", transform=a.transAxes)
            a.axhline(0, c="k", lw=.5); a.set_xlabel("relative layer depth"); a.set_title(lab, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\n  figure -> {path}")


def analyse(paths, plot_path=None, criterion="correct_strict", nperm=2000):
    results = [analyse_one(p, criterion, nperm) for p in paths]
    if len(results) > 1:
        print(f"\n{'model':<34}{'suite':<18}{'acc':>6}{'above':>14}{'PR slope':>11}"
              f"{'p':>7}{'sig layers':>12}")
        for r in results:
            pr = r.get("PR", {}).get("correct")
            print(f"{r['model'].split('/')[-1]:<34}{str(r['suite']):<18}{r['ok'].mean():>6.2f}"
                  f"{str(r['above']):>14}"
                  + (f"{pr['mean']:>+11.2e}{pr['p_mean']:>7.3f}{pr['nsig']:>6}/{r['L']:<5}"
                     if pr else f"{'-':>11}{'-':>7}{'-':>12}"))
    if plot_path:
        plot(results, plot_path)
    return results


# ───────────────────────────── layerwise view ───────────────────────────────
# How each layer's activation changes as serial steps increase.  Per item and
# layer, from the per-head contributions at the measured position:
#   mass   the layer's attention-output mass (sum of head norms)
#   share  that mass as a fraction of the item's total over all layers
#   PR     how evenly the layer's heads share it (participation ratio / H)
# Each line is the mean over items at one depth D, drawn as the change from the
# lowest depth shown, so a layer doing more work at higher D rises with colour.

LAYER_METRICS = {
    "mass":  ("attention-output mass", "% change vs D{b}"),
    "share": ("share of total mass",   "change vs D{b} (pp)"),
    "PR":    ("head dispersion PR/H",  "change vs D{b}"),
}


def layer_profiles(path, metric="mass", correct_only=False,
                   criterion="correct_strict", min_items=5):
    """Per-depth mean layer profiles, as change from the lowest depth kept.

    correct_only keeps correct items at depths where accuracy beats chance and
    at least min_items are correct; otherwise every item is used.
    """
    x = _load(path, criterion)
    D, ok, Y = x["D"], x["ok"], x[metric]
    if correct_only:
        keep = [k for k in _above_chance(D, ok, x["chance"]) if ok[D == k].sum() >= min_items]
        sel = ok
    else:
        keep, sel = sorted(int(k) for k in np.unique(D)), np.ones(len(D), bool)
    prof = {k: Y[sel & (D == k)].mean(0) for k in keep}
    out = dict(model=x["model"], suite=x["suite"], L=x["L"], metric=metric,
               correct_only=correct_only, depths=keep,
               n={k: int((sel & (D == k)).sum()) for k in keep},
               acc=float(ok.mean()), chance=x["chance"], change={}, slope=None)
    if len(keep) < 2:
        return out
    b = prof[keep[0]]
    for k in keep:
        out["change"][k] = (100 * (prof[k] / b - 1) if metric == "mass" else
                            100 * (prof[k] - b) if metric == "share" else prof[k] - b)
    ks = np.array(keep, float); kc = ks - ks.mean()
    Z = np.stack([out["change"][k] for k in keep])
    out["slope"] = kc @ (Z - Z.mean(0)) / (kc @ kc)     # change per extra step, per layer
    return out


def _short(model):
    return model.split("/")[-1]


def layerwise(paths, out_png, metric="mass", correct_only=False,
              criterion="correct_strict", ncols=4, dmax=6):
    """Small multiples: one panel per file, one line per depth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    P = [layer_profiles(p, metric, correct_only, criterion) for p in paths]
    n = len(P); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows),
                             squeeze=False, constrained_layout=True)
    cmap, norm = plt.get_cmap("viridis"), Normalize(1, dmax)
    name, unit = LAYER_METRICS[metric]
    for a, r in zip(axes.flat, P):
        title = f"{_short(r['model'])}\n{r['suite']}  acc {r['acc']:.2f}"
        a.set_title(title, fontsize=8)
        a.axhline(0, c="k", lw=.5)
        if not r["change"]:
            a.text(.5, .5, "fewer than 2 depths\nwith enough correct items",
                   ha="center", va="center", transform=a.transAxes, fontsize=8)
            a.set_xticks([]); a.set_yticks([])
            continue
        xs = np.linspace(0, 1, r["L"])
        for k in r["depths"][1:]:
            a.plot(xs, r["change"][k], c=cmap(norm(k)), lw=1.3)
        a.set_ylabel(unit.format(b=r["depths"][0]), fontsize=7)
        a.tick_params(labelsize=7)
    for a in axes.flat[n:]:
        a.axis("off")
    for a in axes[-1]:
        a.set_xlabel("relative layer depth (0 = first, 1 = last)", fontsize=7)
    sub = "correct items at above-chance depths" if correct_only else "all items"
    fig.suptitle(f"{name} by layer as serial steps increase ({sub}); "
                 f"colour = D", fontsize=11)
    fig.colorbar(ScalarMappable(norm, cmap), ax=axes, shrink=.6, label="D (steps)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return P


def overview(paths, out_png, metric="mass", correct_only=False,
             criterion="correct_strict", bins=40):
    """One row per file: each layer's change per extra step, on a common
    relative-depth axis.  Rows are scaled to their own maximum so the location
    of the effect is comparable across models; the scale is printed at right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = [layer_profiles(p, metric, correct_only, criterion) for p in paths]
    grid = np.full((len(P), bins), np.nan); scale = []
    xb = np.linspace(0, 1, bins)
    for i, r in enumerate(P):
        if r["slope"] is None:
            scale.append(None); continue
        v = np.interp(xb, np.linspace(0, 1, r["L"]), r["slope"])
        m = np.abs(v).max() or 1.0
        grid[i] = v / m; scale.append(m)
    fig, a = plt.subplots(figsize=(10, 0.32 * len(P) + 1.6))
    im = a.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap="RdBu_r",
                  vmin=-1, vmax=1, extent=(0, 1, len(P) - .5, -.5))
    a.set_yticks(range(len(P)))
    a.set_yticklabels([f"{_short(r['model'])} ({r['suite']})" for r in P], fontsize=7)
    for i, m in enumerate(scale):
        a.text(1.01, i, "no data" if m is None else f"±{m:.2g}", va="center",
               fontsize=6, transform=a.get_yaxis_transform())
    name, unit = LAYER_METRICS[metric]
    a.set_xlabel("relative layer depth")
    sub = "correct items, above-chance depths" if correct_only else "all items"
    a.set_title(f"{name}: change per extra step ({unit.split(' vs')[0]} per step, "
                f"row-scaled; {sub})", fontsize=9)
    fig.colorbar(im, ax=a, shrink=.8, label="red = grows with D, blue = shrinks")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--suite", default="serial_depth_fs")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--hf_token", default=None)
    ap.add_argument("--out", default="probe")
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"],
                    help="attention kernel; eager matches earlier runs")
    ap.add_argument("--analyse", nargs="+", metavar="JSON")
    ap.add_argument("--plot", help="write a figure here when analysing")
    ap.add_argument("--criterion", default="correct_strict",
                    choices=["correct_strict", "correct_constrained",
                             "correct_first_digit", "correct"],
                    help="which correctness field selects the items")
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--layerwise", metavar="PNG",
                    help="with --analyse: per-model layer profiles by depth")
    ap.add_argument("--overview", metavar="PNG",
                    help="with --analyse: one heatmap row per file")
    ap.add_argument("--metric", default="mass", choices=sorted(LAYER_METRICS))
    ap.add_argument("--correct-only", action="store_true",
                    help="layerwise/overview on correct items at above-chance depths")
    a = ap.parse_args()
    if a.analyse and (a.layerwise or a.overview):
        if a.layerwise: layerwise(a.analyse, a.layerwise, a.metric, a.correct_only, a.criterion)
        if a.overview:  overview(a.analyse, a.overview, a.metric, a.correct_only, a.criterion)
    elif a.analyse: analyse(a.analyse, a.plot, a.criterion, a.nperm)
    elif a.model: run(a)
    else: ap.error("--model or --analyse required")
