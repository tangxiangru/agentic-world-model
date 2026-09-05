"""Uniform weight average of two full fine-tunes of the same base checkpoint.

Both parents start from the identical gemma-3-4b-pt snapshot and differ only in
training data, so they sit in the same loss basin and averaging is defined.
Output is written as a ready-to-serve directory (bf16 + processor sidecars +
greedy generation config), the same shape finalize_ckpt.py produces.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finalize_ckpt import GEN_CONFIG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="two or more checkpoints of the same base, averaged uniformly")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="mixing weights, one per model; default uniform. Normalised to sum 1.")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--base", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    args = ap.parse_args()

    n = len(args.models)
    if args.weights:
        assert len(args.weights) == n, "one weight per model"
        tot = sum(args.weights)
        ws = [x / tot for x in args.weights]
    else:
        ws = [1.0 / n] * n
    w = ws[0]
    ma = AutoModelForCausalLM.from_pretrained(args.models[0], dtype=torch.float32)
    sa = ma.state_dict()
    n_mixed = 0
    for k in sa:
        if sa[k].is_floating_point():
            sa[k].mul_(w)
            n_mixed += 1
    for path, w in zip(args.models[1:], ws[1:]):
        mb = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        sb = mb.state_dict()
        assert set(sa) == set(sb), "state dicts differ in structure"
        for k in sa:
            if sa[k].is_floating_point():
                sa[k].add_(sb[k], alpha=w)
        del mb, sb
    ma.load_state_dict(sa)
    print(f"averaged {n_mixed} float tensors over {n} checkpoints at weights {[round(x,4) for x in ws]}")
    os.makedirs(args.dst, exist_ok=True)
    ma.config.use_cache = True
    # the parents ship a greedy generation_config (do_sample=False + temperature 0)
    # which save_pretrained refuses to serialise; write a neutral one and replace
    # the file below.
    ma.generation_config.do_sample = True
    ma.generation_config.temperature = 1.0
    ma.generation_config.top_p = 1.0
    ma.generation_config.top_k = 50
    ma.to(torch.bfloat16).save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.dst)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        p = os.path.join(args.base, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.dst, fn))
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(GEN_CONFIG, f, indent=2)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
