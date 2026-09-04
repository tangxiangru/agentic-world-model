#!/usr/bin/env python3
"""Uniform weight soup of two checkpoints on the same training trajectory,
then score it under the session protocol.

exp-03/final is exp-02/final plus one more training stage, so averaging the two
is the classic stochastic-weight-averaging setting: two points on one SGD path.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TASK = "/home/ben/task"
GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "transformers_version": "4.50.0.dev0",
}


def make_soup(a: str, b: str, dst: str, w: float) -> None:
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(a, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        ta = load_file(os.path.join(a, shard))
        tb = load_file(os.path.join(b, shard))
        assert set(ta) == set(tb), shard
        out = {}
        for k in ta:
            x, y = ta[k], tb[k]
            if x.is_floating_point():
                out[k] = (w * x.float() + (1.0 - w) * y.float()).to(x.dtype)
            else:
                assert torch.equal(x, y), k
                out[k] = x
        save_file(out, os.path.join(dst, shard), metadata={"format": "pt"})
        print(f"[soup] {shard}: {len(out)} tensors", flush=True)
    shutil.copy(os.path.join(a, "model.safetensors.index.json"), dst)
    cfg = json.load(open(os.path.join(a, "config.json")))
    cfg["dtype"] = cfg["torch_dtype"] = "bfloat16"
    for k in ("text_config", "vision_config"):
        if k in cfg:
            cfg[k]["dtype"] = cfg[k]["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(dst)
    for f in ("preprocessor_config.json", "processor_config.json"):
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, f))
    json.dump(GREEDY, open(os.path.join(dst, "generation_config.json"), "w"), indent=2)
    print(f"[soup] wrote {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--weight-a", type=float, default=0.5)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max-connections", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    make_soup(args.a, args.b, args.dst, args.weight_a)

    out = f"{TASK}/eval/{args.tag}_dev{args.limit}.json"
    log = f"{TASK}/logs/{args.tag}_dev{args.limit}.log"
    cmd = [
        sys.executable, "evaluate.py",
        "--model-path", args.dst,
        "--limit", str(args.limit),
        "--max-connections", str(args.max_connections),
        "--max-tokens", str(args.max_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--json-output-file", out,
    ]
    print(f"[soup] {' '.join(cmd)}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=TASK, stdout=f, stderr=subprocess.STDOUT)
    print(f"[soup] eval rc={rc}", flush=True)
    if rc == 0 and os.path.exists(out):
        print(f"[soup] {args.tag}: {json.load(open(out))}", flush=True)


if __name__ == "__main__":
    main()
