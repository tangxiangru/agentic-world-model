#!/usr/bin/env python3
"""Uniform weight average ("model soup") of two or more checkpoints with identical shapes.

Writes a directory the grader can load directly: averaged safetensors shards, the
tokenizer/processor files copied from the first input, and a generation_config that asks
vLLM for greedy decoding by name (temperature 0.0 - see exp-02/exp-03).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
COPY = ["config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json",
        "processor_config.json", "chat_template.json"]


def shards(d: str) -> list[str]:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        return sorted(set(json.load(open(idx))["weight_map"].values()))
    return ["model.safetensors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    w = a.weights or [1.0 / len(a.inputs)] * len(a.inputs)
    assert len(w) == len(a.inputs)
    s = sum(w)
    w = [x / s for x in w]
    print("inputs:", list(zip(a.inputs, w)))

    os.makedirs(a.out, exist_ok=True)
    files = shards(a.inputs[0])
    for d in a.inputs[1:]:
        assert shards(d) == files, f"shard layout differs: {d}"

    for fn in files:
        acc: dict[str, torch.Tensor] = {}
        for d, wi in zip(a.inputs, w):
            sd = load_file(os.path.join(d, fn))
            for k, v in sd.items():
                x = v.to(torch.float32) * wi
                acc[k] = x if k not in acc else acc[k] + x
            del sd
        save_file({k: v.to(torch.bfloat16) for k, v in acc.items()},
                  os.path.join(a.out, fn), metadata={"format": "pt"})
        print("wrote", fn, len(acc), "tensors")
        del acc

    idx = os.path.join(a.inputs[0], "model.safetensors.index.json")
    if os.path.exists(idx):
        shutil.copy(idx, a.out)
    for fn in COPY:
        for src in (a.inputs[0], BASE):
            p = os.path.join(src, fn)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(a.out, fn))
                break

    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    for k in ("do_sample", "top_k", "top_p"):
        gc.pop(k, None)
    gc["temperature"] = 0.0
    json.dump(gc, open(os.path.join(a.out, "generation_config.json"), "w"), indent=2)
    print("soup written to", a.out)


if __name__ == "__main__":
    main()
