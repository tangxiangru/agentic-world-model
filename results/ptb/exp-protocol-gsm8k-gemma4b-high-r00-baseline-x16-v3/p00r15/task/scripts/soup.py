#!/usr/bin/env python3
"""Uniformly average the weights of several checkpoints from the same trajectory.

exp-05 was initialised from exp-04, so these are points on one optimisation path;
averaging along a path is the cheapest variance-reduction there is (no GPU, no data).
Tokenizer/config/generation_config are taken from the last checkpoint listed.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acc = None
    for i, c in enumerate(args.ckpts):
        print(f"[soup] loading {c}", flush=True)
        m = Gemma3ForConditionalGeneration.from_pretrained(c, dtype=torch.float32, device_map="cpu")
        sd = m.state_dict()
        if acc is None:
            acc = {k: v.clone() for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k]
        del m, sd

    n = len(args.ckpts)
    for k in acc:
        acc[k] /= n
        acc[k] = acc[k].to(torch.bfloat16)

    print(f"[soup] writing {args.out}", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.ckpts[-1], dtype=torch.bfloat16, device_map="cpu")
    missing, unexpected = model.load_state_dict(acc, strict=False)
    assert not unexpected, unexpected
    if missing:
        print("[soup] missing (tied weights are expected here):", missing)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.ckpts[-1]).save_pretrained(args.out)
    for fn in ["generation_config.json", "preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(args.ckpts[-1], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    print("[soup] done", flush=True)


if __name__ == "__main__":
    main()
