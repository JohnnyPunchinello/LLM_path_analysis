#!/usr/bin/env python3
"""
Can the contribution metric run on this model?  Answers from the config alone.

The metric reads the input to `attn.o_proj` and reshapes it to
(n_heads, d_head), so it requires an attention module exposing `o_proj` whose
input width equals n_heads * d_head.  Architectures that name the projection
differently (GPT-2 `c_proj`, NeoX/Falcon `dense`, MPT `out_proj`) fall back to
the attribution metric, which -- per the manuscript -- disagrees with
contribution at the layer level and is therefore not interchangeable with it.
Latent-attention designs (DeepSeek MLA) expose `o_proj` but do not decompose
into independent per-head slices, so the reshape is invalid.

Builds the module tree on the meta device: no weights are downloaded, so a
100B-parameter model is checked in seconds.

    python3 check_compat.py meta-llama/Llama-3.1-70B mistralai/Mistral-7B-v0.1 ...
    python3 check_compat.py --preset large
"""
import argparse, sys

PRESETS = {
    "predicted": [   # families the locality literature groups with Llama -> predict a band
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-Small-24B-Base-2501",
    ],
    "contrast": [    # architecturally distinct designs
        "google/gemma-2-9b", "google/gemma-2-27b",
        "microsoft/Phi-3-medium-4k-instruct",
        "01-ai/Yi-1.5-34B",
        "CohereForAI/c4ai-command-r-v01",
    ],
    "large": [       # <= ~80B, reachable in bf16 with CPU offload on one 80GB card
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-32B",
        "google/gemma-2-27b",
        "01-ai/Yi-1.5-34B",
        "mistralai/Mixtral-8x7B-v0.1",
    ],
}


def check(repo, token=None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    row = {"repo": repo}
    try:
        cfg = AutoConfig.from_pretrained(repo, token=token, trust_remote_code=False)
    except Exception as e:
        row["verdict"] = f"config failed: {type(e).__name__}"
        return row
    row["arch"] = (getattr(cfg, "architectures", None) or ["?"])[0]
    for k, a in (("L", "num_hidden_layers"), ("H", "num_attention_heads"),
                 ("KV", "num_key_value_heads"), ("d", "hidden_size")):
        row[k] = getattr(cfg, a, None)
    try:
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:
        row["verdict"] = f"build failed: {type(e).__name__}"
        return row

    layers = None
    for path in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        o = m
        for p in path:
            o = getattr(o, p, None)
            if o is None: break
        if o is not None:
            layers = o; break
    if layers is None:
        row["verdict"] = "no layer list found"; return row

    attn = None
    for n in ("self_attn", "attention", "attn"):
        if hasattr(layers[0], n): attn = getattr(layers[0], n); break
    if attn is None:
        row["verdict"] = "no attention module"; return row
    row["attn_cls"] = type(attn).__name__

    op = getattr(attn, "o_proj", None)
    if op is None:
        alt = [n for n in ("c_proj", "dense", "out_proj", "wo") if hasattr(attn, n)]
        row["verdict"] = (f"no o_proj (found {alt[0]}) -> attribution only"
                          if alt else "no output projection found")
        return row

    in_f = getattr(op, "in_features", None)
    H, d = row.get("H"), row.get("d")
    hd = getattr(cfg, "head_dim", None) or (d // H if (H and d) else None)
    row["o_proj_in"] = in_f
    row["head_dim"] = hd
    if in_f and H and hd and in_f == H * hd:
        # The model is fine, but the pipeline must use cfg.head_dim rather than
        # hidden_size // n_heads.  When those differ, older revisions reshape to
        # the wrong width, the exception is swallowed, and scores come back NaN.
        if d and hd != d // H:
            row["verdict"] = (f"OK - contribution, BUT head_dim={hd} != hidden/H={d//H}; "
                              "requires the head_dim fix in load_model")
        else:
            row["verdict"] = "OK - contribution"
    elif in_f and H and hd:
        row["verdict"] = f"o_proj in={in_f} != H*head_dim={H*hd} -> check head layout"
    else:
        row["verdict"] = "o_proj present, shape unverified"
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--token", default=None)
    a = ap.parse_args()
    repos = list(a.repos) + (PRESETS[a.preset] if a.preset else [])
    if not repos:
        ap.error("give repo ids or --preset " + "/".join(sorted(PRESETS)))
    print(f"{'repo':<42}{'L':>4}{'H':>4}{'KV':>4}{'d':>6}  {'attention class':<26} verdict")
    print("-" * 122)
    for r in repos:
        x = check(r, a.token)
        print(f"{x['repo']:<42}{str(x.get('L','-')):>4}{str(x.get('H','-')):>4}"
              f"{str(x.get('KV','-')):>4}{str(x.get('d','-')):>6}  "
              f"{str(x.get('attn_cls','-')):<26} {x['verdict']}")
