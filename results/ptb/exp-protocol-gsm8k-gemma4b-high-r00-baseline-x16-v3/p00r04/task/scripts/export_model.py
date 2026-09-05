"""Copy a training checkpoint into a directory the grader can load: bf16 weights,
tokenizer + processor alongside, config.torch_dtype fixed.

The grader runs `evaluate.py --model-path final_model` with
gpu_memory_utilization=0.3 (~24 GB), so an fp32 copy of a 4 B model would not
fit; intermediate Trainer checkpoints are fp32 and must be cast here.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import fmt as F  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tokenizer-from", default=F.SNAPSHOT)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = "bfloat16"
    model.config.use_cache = True
    os.makedirs(args.dst, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.tokenizer_from).save_pretrained(args.dst)
    try:
        AutoProcessor.from_pretrained(args.tokenizer_from).save_pretrained(args.dst)
    except Exception as e:
        print("processor save skipped:", e)
    print("wrote", args.dst)
    print("files:", sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
