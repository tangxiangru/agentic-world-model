"""Uniform weight average ("model soup") of two or more checkpoints of the same shape.

CPU only, so it can be built while the GPU is busy. Writes a full servable directory:
bf16 weights, the aux/tokenizer files copied from the first input, and config.json
dtype forced to bfloat16 the same way scripts/train_sft.py:save_full does.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

AUX = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "special_tokens_map.json", "tokenizer.json",
    "tokenizer.model", "tokenizer_config.json", "added_tokens.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    idx_path = os.path.join(args.inputs[0], "model.safetensors.index.json")
    index = json.load(open(idx_path))
    shards = sorted(set(index["weight_map"].values()))

    for shard in shards:
        acc = None
        for i, d in enumerate(args.inputs):
            sd = load_file(os.path.join(d, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) for k, v in sd.items()}
            else:
                assert set(sd) == set(acc), f"key mismatch in {shard} for {d}"
                for k in acc:
                    acc[k] += sd[k].to(torch.float32)
            del sd
        n = len(args.inputs)
        out = {k: (v / n).to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
        print("[soup]", shard, len(out), "tensors")
        del acc, out

    shutil.copy(idx_path, os.path.join(args.out, "model.safetensors.index.json"))
    for name in AUX:
        s = os.path.join(args.inputs[0], name)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(args.out, name))
    cfg_path = os.path.join(args.out, "config.json")
    cfg = json.load(open(cfg_path))
    for key in ("torch_dtype", "dtype"):
        cfg[key] = "bfloat16"
    for sub in ("text_config", "vision_config"):
        if isinstance(cfg.get(sub), dict):
            for key in ("torch_dtype", "dtype"):
                if key in cfg[sub]:
                    cfg[sub][key] = "bfloat16"
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print("[soup] wrote", args.out, "from", len(args.inputs), "inputs")


if __name__ == "__main__":
    main()
