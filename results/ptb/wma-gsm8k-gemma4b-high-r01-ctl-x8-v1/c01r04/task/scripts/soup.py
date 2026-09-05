#!/usr/bin/env python3
"""Uniform weight average of several checkpoints of the same architecture (model soup).

Streams the safetensors shards one tensor at a time on CPU so it never holds more than
one full model in memory, then writes the result with the same shard layout.
"""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = a.weights or [1.0 / len(a.ckpts)] * len(a.ckpts)
    assert len(w) == len(a.ckpts)
    tot = sum(w)
    w = [x / tot for x in w]
    print("averaging", list(zip(a.ckpts, w)))

    base = a.ckpts[0]
    os.makedirs(a.out, exist_ok=True)
    for fn in os.listdir(base):
        if not fn.endswith(".safetensors"):
            shutil.copy(os.path.join(base, fn), os.path.join(a.out, fn))

    idx_path = os.path.join(base, "model.safetensors.index.json")
    shards = (sorted(set(json.load(open(idx_path))["weight_map"].values()))
              if os.path.exists(idx_path)
              else [f for f in os.listdir(base) if f.endswith(".safetensors")])

    for shard in shards:
        acc = None
        for ck, wi in zip(a.ckpts, w):
            sd = load_file(os.path.join(ck, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) * wi for k, v in sd.items()}
            else:
                assert set(sd) == set(acc), f"key mismatch in {ck}/{shard}"
                for k in acc:
                    acc[k] += sd[k].to(torch.float32) * wi
            del sd
        out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(a.out, shard), metadata={"format": "pt"})
        print("wrote", shard, len(out), "tensors", flush=True)
    print("soup at", a.out)


if __name__ == "__main__":
    main()
