#!/usr/bin/env python3
"""Uniform (or weighted) parameter average of checkpoints that share an init.

All ingredients must come from the same base checkpoint, or the average is
meaningless. Writes a directory package_final.py can stage for evaluation.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

BASE = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = args.weights or [1.0] * len(args.ckpts)
    assert len(w) == len(args.ckpts)
    tot = sum(w)
    w = [x / tot for x in w]
    print("ingredients:", list(zip(args.ckpts, w)))

    acc = None
    for path, wi in zip(args.ckpts, w):
        m = Gemma3ForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float32, device_map="cpu"
        )
        sd = m.state_dict()
        if acc is None:
            acc = {k: v.clone() * wi for k, v in sd.items()}
        else:
            assert set(acc) == set(sd), "checkpoints have different parameter sets"
            for k in acc:
                acc[k] += sd[k] * wi
        del m, sd
        print("folded", path, flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.ckpts[0], torch_dtype=torch.float32, device_map="cpu"
    )
    model.load_state_dict(acc)
    model = model.to(torch.bfloat16)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.out)
    gp = os.path.join(args.out, "generation_config.json")
    gc = json.load(open(gp)) if os.path.exists(gp) else {}
    gc["eos_token_id"] = [1, 106]
    gc["bos_token_id"] = 2
    gc["pad_token_id"] = 0
    json.dump(gc, open(gp, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
