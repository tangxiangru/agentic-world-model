#!/usr/bin/env python3
"""Turn a Trainer checkpoint-N/ directory into a vLLM-loadable bf16 model dir.

Mid-training checkpoints are fp32 (the run keeps fp32 master weights) and carry
no tokenizer and no generation_config, so evaluate.py cannot read them as they
stand. This writes the artifact the grader actually loads.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import SNAPSHOT
from train_sft import write_generation_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tokenizer", default=SNAPSHOT)
    args = ap.parse_args()

    # save_pretrained stamps config.dtype from the model's real dtype in 4.57,
    # so loading as bf16 is all that is needed to un-fp32 a Trainer checkpoint
    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16)
    model.config.use_cache = True
    os.makedirs(args.dst, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.tokenizer).save_pretrained(args.dst)
    for extra in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAPSHOT, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.dst, extra))
    gen = write_generation_config(args.dst)
    print(json.dumps({"dst": args.dst, "generation_config": gen}, indent=2))


if __name__ == "__main__":
    main()
