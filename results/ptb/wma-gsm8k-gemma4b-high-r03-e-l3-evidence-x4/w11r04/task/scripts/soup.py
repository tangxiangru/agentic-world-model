#!/usr/bin/env python3
"""Uniform (or weighted) parameter average of two or more checkpoints.

All ingredients must be the same architecture and come from the same base
snapshot.  Output is written bf16 with the greedy generation_config and the
parent snapshot's tokenizer/processor files, i.e. directly servable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoConfig, AutoTokenizer

from prep_ckpt import COPY_FILES, GEN_CFG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="path[:weight]")
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    specs = []
    for s in args.ckpt:
        path, _, w = s.rpartition(":")
        if not path or not w.replace(".", "").isdigit():
            path, w = s, "1"
        specs.append((path, float(w)))
    total = sum(w for _, w in specs)
    print(specs, "total weight", total)

    cfg = AutoConfig.from_pretrained(specs[0][0])
    is_mm = cfg.architectures and "ConditionalGeneration" in cfg.architectures[0]
    if is_mm:
        from transformers import Gemma3ForConditionalGeneration as M
    else:
        from transformers import AutoModelForCausalLM as M

    base = None
    for path, w in specs:
        m = M.from_pretrained(path, dtype=torch.float32)
        sd = m.state_dict()
        if base is None:
            base = {k: v * (w / total) for k, v in sd.items()}
            model = m
        else:
            for k in base:
                base[k] += sd[k] * (w / total)
            del m
        del sd
    model.load_state_dict(base)
    model = model.to(torch.bfloat16)
    model.config.use_cache = True
    model.generation_config = type(model.generation_config).from_dict(GEN_CFG)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.parent)
    tok.save_pretrained(args.out)
    for f in COPY_FILES:
        src = os.path.join(args.parent, f)
        dst = os.path.join(args.out, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(GEN_CFG, f, indent=2)
    print("souped ->", args.out)


if __name__ == "__main__":
    main()
