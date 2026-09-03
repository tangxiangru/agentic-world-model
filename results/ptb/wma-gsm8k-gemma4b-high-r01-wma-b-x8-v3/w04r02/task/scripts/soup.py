#!/usr/bin/env python3
"""Uniform weight average (model soup) of several checkpoints of the same model.

Runs on CPU. Writes a full checkpoint dir: averaged safetensors shards with the
same sharding as the first source, plus the small config/tokenizer files.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--extra-from", required=True, help="dir with tokenizer/processor/config files")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.srcs[0], "model.safetensors.index.json")
    index = json.load(open(index_path))
    shards = sorted(set(index["weight_map"].values()))
    print("shards:", shards, "srcs:", len(args.srcs))

    for shard in shards:
        acc: dict[str, torch.Tensor] = {}
        for i, src in enumerate(args.srcs):
            sd = load_file(os.path.join(src, shard))
            for k, v in sd.items():
                f = v.to(torch.float32)
                acc[k] = f if i == 0 else acc[k] + f
            del sd
        n = len(args.srcs)
        out = {k: (v / n).to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.dst, shard), metadata={"format": "pt"})
        print("wrote", shard, len(out), "tensors", flush=True)
        del acc, out

    shutil.copy2(index_path, os.path.join(args.dst, "model.safetensors.index.json"))
    for name in os.listdir(args.extra_from):
        s, d = os.path.join(args.extra_from, name), os.path.join(args.dst, name)
        if os.path.exists(d) or os.path.isdir(s):
            continue
        if name.endswith((".safetensors", ".bin", ".pt")) or name == "model.safetensors.index.json":
            continue
        shutil.copy2(s, d)
    print("soup written to", args.dst)


if __name__ == "__main__":
    main()
