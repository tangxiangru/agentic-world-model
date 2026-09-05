#!/usr/bin/env python3
"""Uniform weight average of checkpoints from one training trajectory.

Same architecture, same tokenizer, no training. Writes a vLLM-ready bf16 dir.
"""
from __future__ import annotations

import argparse
import json
import shutil
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import SNAPSHOT
from train_sft import write_generation_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=SNAPSHOT)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.inputs[0], dtype=torch.float32)
    acc = {k: v.clone() for k, v in model.state_dict().items()}
    for extra in args.inputs[1:]:
        other = AutoModelForCausalLM.from_pretrained(extra, dtype=torch.float32)
        for k, v in other.state_dict().items():
            acc[k] += v
        del other
    n = len(args.inputs)
    for k in acc:
        if acc[k].is_floating_point():
            acc[k] /= n
        else:
            acc[k] = acc[k] // n
    model.load_state_dict(acc)
    model = model.to(torch.bfloat16)
    model.config.use_cache = True

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.tokenizer).save_pretrained(args.out)
    for extra in ("preprocessor_config.json", "processor_config.json"):
        shutil.copy(os.path.join(SNAPSHOT, extra), os.path.join(args.out, extra))
    write_generation_config(args.out)
    print(json.dumps({"out": args.out, "ingredients": args.inputs}, indent=2))


if __name__ == "__main__":
    main()
