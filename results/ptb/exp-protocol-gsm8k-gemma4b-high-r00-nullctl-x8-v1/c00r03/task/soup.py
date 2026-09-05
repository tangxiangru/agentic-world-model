#!/usr/bin/env python3
"""Average the weights of several checkpoints fine-tuned from a common init."""
import argparse, glob, json, os, shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = args.models[0]
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(ref):
        if not f.endswith(".safetensors"):
            shutil.copy(os.path.join(ref, f), os.path.join(args.out, f))

    shards = sorted(os.path.basename(p) for p in
                    glob.glob(os.path.join(ref, "*.safetensors")))
    n = len(args.models)
    for sh in shards:
        acc = None
        for m in args.models:
            sd = load_file(os.path.join(m, sh))
            if acc is None:
                acc = {k: v.to(torch.float32) for k, v in sd.items()}
            else:
                for k in acc:
                    acc[k] += sd[k].to(torch.float32)
            del sd
        out = {k: (v / n).to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.out, sh), metadata={"format": "pt"})
        print("wrote", sh, flush=True)
        del acc, out
    print("soup of", n, "->", args.out)


if __name__ == "__main__":
    main()
