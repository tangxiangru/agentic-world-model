"""Uniform weight average ("model soup") of two or more checkpoints of the same
architecture, saved in bf16 with tokenizer and processor so vLLM can load it.
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
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    w = args.weights or [1.0 / len(args.srcs)] * len(args.srcs)
    assert len(w) == len(args.srcs)
    s = sum(w)
    w = [x / s for x in w]

    base = AutoModelForCausalLM.from_pretrained(args.srcs[0], dtype=torch.float32)
    acc = {k: v.clone() * w[0] for k, v in base.state_dict().items()}
    for src, wi in zip(args.srcs[1:], w[1:]):
        m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
        sd = m.state_dict()
        for k in acc:
            acc[k] += sd[k] * wi
        del m, sd
    base.load_state_dict(acc)
    base = base.to(torch.bfloat16)
    base.config.torch_dtype = "bfloat16"
    if hasattr(base.config, "text_config"):
        base.config.text_config.torch_dtype = "bfloat16"
    base.config.use_cache = True
    os.makedirs(args.dst, exist_ok=True)
    base.save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(F.SNAPSHOT).save_pretrained(args.dst)
    try:
        AutoProcessor.from_pretrained(F.SNAPSHOT).save_pretrained(args.dst)
    except Exception as e:
        print("processor save skipped:", e)
    print("wrote", args.dst, "weights", w)


if __name__ == "__main__":
    main()
