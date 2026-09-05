"""Uniformly average the weights of two or more checkpoints of the same shape.

Only useful for checkpoints fine-tuned from the same base: they stay in one
basin, so the average is usually a slightly better model than either. Tokenizer,
config and the processor files are copied from the first checkpoint.
"""
import argparse
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(w) == len(args.models), "one weight per model"
    total = sum(w)
    w = [x / total for x in w]
    print("weights:", dict(zip(args.models, w)), flush=True)

    os.makedirs(args.out, exist_ok=True)
    first = args.models[0]
    shards = sorted(f for f in os.listdir(first) if f.endswith(".safetensors"))
    for shard in shards:
        acc = None
        for m, wi in zip(args.models, w):
            sd = load_file(os.path.join(m, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) * wi for k, v in sd.items()}
            else:
                for k in acc:
                    acc[k] += sd[k].to(torch.float32) * wi
            del sd
        save_file({k: v.to(torch.bfloat16) for k, v in acc.items()},
                  os.path.join(args.out, shard), metadata={"format": "pt"})
        print("wrote", shard, flush=True)
        del acc

    for f in os.listdir(first):
        if f.endswith(".safetensors") or f == "training_args.bin":
            continue
        shutil.copyfile(os.path.join(first, f), os.path.join(args.out, f))
    print("done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
