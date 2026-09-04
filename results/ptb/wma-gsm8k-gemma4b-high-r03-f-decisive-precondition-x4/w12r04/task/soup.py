#!/usr/bin/env python3
"""Weight-average two or more checkpoints of the same architecture (CPU, bf16 out).

Averaging a checkpoint with its own continuation is an EMA over the last leg of
training; it costs no GPU and sometimes recovers a point the final step lost.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.src)] * len(args.src)
    assert len(w) == len(args.src)
    w = [x / sum(w) for x in w]
    print("mixing", list(zip(args.src, w)))

    acc = None
    for path, wi in zip(args.src, w):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        sd = m.state_dict()
        if acc is None:
            acc = {k: v.clone().mul_(wi) for k, v in sd.items()}
            keeper = m
        else:
            for k in acc:
                acc[k].add_(sd[k], alpha=wi)
            del m
    keeper.load_state_dict(acc)
    keeper = keeper.to(torch.bfloat16)

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    keeper.save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.dst)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        shutil.copy(os.path.join(BASE, fn), os.path.join(args.dst, fn))
    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    gc["temperature"] = args.temperature
    gc["do_sample"] = args.temperature > 0
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)

    del keeper, acc
    m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16)
    print("[verify] reloaded", type(m).__name__, sum(p.numel() for p in m.parameters()) / 1e9, "B")
    print("[verify] generation_config", json.load(open(os.path.join(args.dst, "generation_config.json"))))


if __name__ == "__main__":
    main()
