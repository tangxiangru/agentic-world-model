"""Uniformly average the weights of several fine-tunes of the same base model
and write a checkpoint the grader can load (bf16, greedy generation_config).

All inputs must come from the same initialisation, which they do here: every
checkpoint in this session descends from the pinned gemma-3-4b-pt snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def shard_map(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    return idx["weight_map"], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.inputs)] * len(args.inputs)
    assert len(w) == len(args.inputs)
    w = [x / sum(w) for x in w]
    print("[soup]", list(zip(args.inputs, w)), flush=True)

    os.makedirs(args.out, exist_ok=True)
    ref = args.inputs[0]
    wmap, idx = shard_map(ref)
    shards = sorted(set(wmap.values()))

    # load every input's shard lazily, one shard at a time
    caches = [{} for _ in args.inputs]
    for shard in shards:
        acc = None
        for i, d in enumerate(args.inputs):
            m, _ = shard_map(d)
            # a key may live in a different shard file in another checkpoint;
            # collect this shard's keys from wherever they are
            keys = [k for k, v in wmap.items() if v == shard]
            files = sorted({m[k] for k in keys})
            for f in files:
                if f not in caches[i]:
                    caches[i][f] = load_file(os.path.join(d, f))
            part = {k: caches[i][m[k]][k] for k in keys}
            if acc is None:
                acc = {k: part[k].to(torch.float32) * w[i] for k in keys}
            else:
                for k in keys:
                    acc[k] += part[k].to(torch.float32) * w[i]
            caches[i] = {}  # free
        out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
        print("[soup] wrote", shard, len(out), "tensors", flush=True)
        del acc, out

    shutil.copy(os.path.join(ref, "model.safetensors.index.json"),
                os.path.join(args.out, "model.safetensors.index.json"))
    cfg = json.load(open(os.path.join(ref, "config.json")))
    for c in (cfg, cfg.get("text_config", {}), cfg.get("vision_config", {})):
        if isinstance(c, dict):
            for k in ("dtype", "torch_dtype"):
                if k in c or c is cfg:
                    c[k] = "bfloat16"
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    AutoTokenizer.from_pretrained(SNAP).save_pretrained(args.out)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        shutil.copy(os.path.join(SNAP, fn), os.path.join(args.out, fn))
    json.dump({"bos_token_id": 2, "cache_implementation": "hybrid",
               "eos_token_id": [1, 106], "pad_token_id": 0,
               "do_sample": False, "temperature": 0.0,
               "transformers_version": "4.50.0.dev0"},
              open(os.path.join(args.out, "generation_config.json"), "w"), indent=2)
    print("[soup] done ->", args.out)


if __name__ == "__main__":
    main()
