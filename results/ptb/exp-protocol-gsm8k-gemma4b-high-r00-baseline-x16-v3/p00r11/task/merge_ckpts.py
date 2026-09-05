#!/usr/bin/env python3
"""Uniform weight average of two checkpoints of the same architecture."""
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
    ap.add_argument("--weight-a", type=float, default=0.5)
    args = ap.parse_args()

    ma = Gemma3ForConditionalGeneration.from_pretrained(args.a, dtype=torch.bfloat16)
    mb = Gemma3ForConditionalGeneration.from_pretrained(args.b, dtype=torch.bfloat16)
    sa, sb = ma.state_dict(), mb.state_dict()
    assert set(sa) == set(sb), "checkpoints have different parameter sets"
    w = args.weight_a
    merged = {k: (sa[k].float() * w + sb[k].float() * (1 - w)).to(torch.bfloat16)
              for k in sa}
    ma.load_state_dict(merged)
    ma.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.a).save_pretrained(args.out)
    for fn in ("preprocessor_config.json", "processor_config.json",
               "generation_config.json"):
        src = os.path.join(args.a, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(args.out, fn)):
            shutil.copy(src, os.path.join(args.out, fn))
    print("[merge] wrote", args.out)


if __name__ == "__main__":
    main()
