#!/usr/bin/env python3
"""Uniform weight average ("soup") of two checkpoints on the same trajectory."""
from __future__ import annotations

import argparse
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
    args = ap.parse_args()

    print("loading", args.a)
    ma = Gemma3ForConditionalGeneration.from_pretrained(args.a, dtype=torch.bfloat16, device_map="cpu")
    print("loading", args.b)
    mb = Gemma3ForConditionalGeneration.from_pretrained(args.b, dtype=torch.bfloat16, device_map="cpu")

    sa, sb = ma.state_dict(), mb.state_dict()
    assert set(sa) == set(sb), "state dicts differ in keys"
    n_diff = 0
    with torch.no_grad():
        for k in sa:
            if sa[k].dtype.is_floating_point:
                if not torch.equal(sa[k], sb[k]):
                    n_diff += 1
                sa[k].mul_(args.alpha).add_(sb[k].to(sa[k].dtype), alpha=1.0 - args.alpha)
    print(f"averaged {len(sa)} tensors ({n_diff} actually differed), alpha={args.alpha}")
    ma.load_state_dict(sa)
    os.makedirs(args.out, exist_ok=True)
    ma.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.a).save_pretrained(args.out)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.a, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, f))
    print("saved", args.out)


if __name__ == "__main__":
    main()
