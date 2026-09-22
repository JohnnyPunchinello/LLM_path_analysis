#!/usr/bin/env python3
"""
Can the contribution metric run on this model?  Answers from the config alone.

The metric reads the input to the attention output projection and reshapes it
to (n_heads, d_head), so it requires a projection whose input width equals
n_heads * d_head.  deep_probe.py hooks `o_proj`, `dense` (Pythia/NeoX),
`c_proj` (GPT-2, a Conv1D) and `out_proj`/`wo`, and verifies the weight layout
functionally on the first forward, so all of these use the same contribution
metric -- never the gradient-based attribution fallback.  Latent-attention
designs (DeepSeek MLA) expose `o_proj` but do not decompose into independent
per-head slices; deep_probe refuses those at run time.

--align SUITE additionally loads each tokenizer (no weights) and reports where
the activation will be measured: the position whose next token is the answer
digit (see deep_probe.align_answer).

Builds the module tree on the meta device: no weights are downloaded, so a
100B-parameter model is checked in seconds.

    python3 check_compat.py meta-llama/Llama-3.1-70B mistralai/Mistral-7B-v0.1 ...
    python3 check_compat.py --preset large
    python3 check_compat.py gpt2 Qwen/Qwen2.5-0.5B --align serial_depth_fs
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

    op, pname = None, None
    for n in ("o_proj", "dense", "c_proj", "out_proj", "wo"):   # deep_probe's order
        if hasattr(attn, n) and hasattr(getattr(attn, n), "weight"):
            op, pname = getattr(attn, n), n
            break
    if op is None:
        row["verdict"] = "no output projection found"
        return row
    row["proj"] = pname

    # nn.Linear has in_features; GPT-2's Conv1D stores weight as [in, out]
    in_f = getattr(op, "in_features", None) or (
        op.weight.shape[0] if type(op).__name__ == "Conv1D" else None)
    H, d = row.get("H"), row.get("d")
    hd = getattr(cfg, "head_dim", None) or (d // H if (H and d) else None)
    row["o_proj_in"] = in_f
    row["head_dim"] = hd
    if in_f and H and hd and in_f == H * hd:
        # The model is fine, but the pipeline must use cfg.head_dim rather than
        # hidden_size // n_heads.  When those differ, older revisions reshape to
        # the wrong width, the exception is swallowed, and scores come back NaN.
        via = "" if pname == "o_proj" else f" via {pname}"
        if d and hd != d // H:
            row["verdict"] = (f"OK - contribution{via}, head_dim={hd} != hidden/H={d//H} "
                              "(handled by the head_dim fix in load_model)")
        else:
            row["verdict"] = f"OK - contribution{via}"
    elif in_f and H and hd:
        row["verdict"] = f"o_proj in={in_f} != H*head_dim={H*hd} -> check head layout"
    else:
        row["verdict"] = "o_proj present, shape unverified"
    return row


def check_align(repo, suite, token=None):
    """Where will deep_probe measure?  Tokenizer only, every prompt of the suite."""
    try:
        from transformers import AutoTokenizer
        import active_subgraph_dot as asd
        from deep_probe import align_answer
        tok = AutoTokenizer.from_pretrained(repo, token=token)
        forms = set()
        for text in asd.TASK_SUITES[suite]["tasks"]:
            forms.add(tuple(align_answer(tok, text)[2]))
    except SystemExit as e:
        return f"ALIGN FAILED: {e}"
    except Exception as e:
        return f"tokenizer failed: {type(e).__name__}: {str(e)[:80]}"
    if len(forms) > 1:
        return f"ALIGN FAILED: inconsistent across prompts {sorted(forms)}"
    extra = next(iter(forms))
    return "last prompt token" if not extra else f"prompt + {list(extra)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--token", default=None)
    ap.add_argument("--align", metavar="SUITE",
                    help="also check answer alignment on this suite's prompts")
    a = ap.parse_args()
    repos = list(a.repos) + (PRESETS[a.preset] if a.preset else [])
    if not repos:
        ap.error("give repo ids or --preset " + "/".join(sorted(PRESETS)))
    print(f"{'repo':<42}{'L':>4}{'H':>4}{'KV':>4}{'d':>6}  {'attention class':<26} verdict")
    print("-" * 122)
    for r in repos:
        x = check(r, a.token)
        print(f"{x['repo']:<42}{str(x.get('L','-')):>4}{str(x.get('H','-')):>4}"
              f"{str(x.get('KV') or '-'):>4}{str(x.get('d','-')):>6}  "
              f"{str(x.get('attn_cls','-')):<26} {x['verdict']}")
        if a.align:
            print(f"{'':<42}measured position ({a.align}): {check_align(r, a.align, a.token)}")
