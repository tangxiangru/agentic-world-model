#!/usr/bin/env python3
"""Uniform weight average (model soup) of several fine-tunes of the same base."""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Gemma3ForConditionalGeneration.from_pretrained(args.models[0], dtype=torch.float32)
    sd = base.state_dict()
    for k in sd:
        sd[k] = sd[k].float()
    for m in args.models[1:]:
        other = Gemma3ForConditionalGeneration.from_pretrained(m, dtype=torch.float32)
        osd = other.state_dict()
        assert set(osd) == set(sd), "state dicts differ"
        for k in sd:
            sd[k] += osd[k].float()
        del other, osd
    n = len(args.models)
    for k in sd:
        sd[k] = (sd[k] / n).to(torch.bfloat16)
    base.load_state_dict(sd)
    base = base.to(torch.bfloat16)
    base.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(fmt.SNAPSHOT).save_pretrained(args.out)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(fmt.SNAPSHOT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("souped", n, "models ->", args.out)


if __name__ == "__main__":
    main()
