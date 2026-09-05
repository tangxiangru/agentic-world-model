"""Weight-space average of two checkpoints fine-tuned from the same init.

Writes a full, loadable model directory (weights + tokenizer + gemma-3
processor files + the greedy generation_config), so the output can be handed
straight to evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    n_params = 0
    for shard in shards:
        ta = load_file(os.path.join(args.a, shard))
        tb = load_file(os.path.join(args.b, shard))
        assert set(ta) == set(tb), f"key mismatch in {shard}"
        out = {}
        for k, va in ta.items():
            vb = tb[k]
            assert va.shape == vb.shape, f"shape mismatch for {k}"
            out[k] = (
                args.alpha * va.to(torch.float32) + (1 - args.alpha) * vb.to(torch.float32)
            ).to(va.dtype)
            n_params += va.numel()
        save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
        print("merged", shard, flush=True)
        del ta, tb, out

    for f in ("model.safetensors.index.json", "config.json", "generation_config.json",
              "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "added_tokens.json", "tokenizer.model", "preprocessor_config.json",
              "processor_config.json"):
        src = os.path.join(args.a, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, f))
    print(f"souped {n_params/1e9:.2f}B params at alpha={args.alpha} -> {args.out}")


if __name__ == "__main__":
    main()
