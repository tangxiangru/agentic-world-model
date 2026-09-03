"""Uniform weight average of two checkpoints of the same architecture."""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def load_all(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    sd = {}
    for shard in sorted(set(idx["weight_map"].values())):
        sd.update(load_file(os.path.join(path, shard)))
    return sd, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight-a", type=float, default=0.5)
    args = ap.parse_args()

    sa, idx = load_all(args.a)
    sb, _ = load_all(args.b)
    assert set(sa) == set(sb), (len(sa), len(sb))
    w = args.weight_a
    out = {k: (sa[k].float() * w + sb[k].float() * (1 - w)).to(sa[k].dtype) for k in sa}

    os.makedirs(args.out, exist_ok=True)
    shards = {}
    for k, shard in idx["weight_map"].items():
        shards.setdefault(shard, {})[k] = out[k]
    for shard, tensors in shards.items():
        save_file(tensors, os.path.join(args.out, shard), metadata={"format": "pt"})
    for fn in os.listdir(args.a):
        if fn.endswith(".safetensors") or fn == "train_log.jsonl":
            continue
        src = os.path.realpath(os.path.join(args.a, fn))
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, fn))
    print(f"souped {len(out)} tensors, weight_a={w} -> {args.out}")


if __name__ == "__main__":
    main()
