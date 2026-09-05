#!/usr/bin/env python3
"""Uniform weight average of two checkpoints from the same training trajectory.

exp-05 is exp-02's checkpoint continued for one epoch on on-policy data, so the
two sit on one trajectory and their average is checkpoint averaging, not a
merge of unrelated models. Runs on CPU in float32 and saves bf16.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

AUX = ["config.json", "model.safetensors.index.json", "tokenizer.json",
       "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
       "added_tokens.json", "preprocessor_config.json", "processor_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a, b, out = Path(args.a), Path(args.b), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in AUX:
        if (a / f).exists():
            shutil.copy2(a / f, out / f)

    index = json.loads((a / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    n_avg = 0
    for shard in shards:
        ta, tb = load_file(str(a / shard)), load_file(str(b / shard))
        assert ta.keys() == tb.keys(), f"key mismatch in {shard}"
        merged = {}
        for k, va in ta.items():
            vb = tb[k]
            if va.is_floating_point():
                merged[k] = (args.alpha * va.float() + (1 - args.alpha) * vb.float()).to(va.dtype)
                n_avg += 1
            else:
                assert torch.equal(va, vb), f"non-float tensor differs: {k}"
                merged[k] = va
        save_file(merged, str(out / shard), metadata={"format": "pt"})
        print(f"[soup] {shard}: {len(merged)} tensors", flush=True)

    gen = json.loads((a / "generation_config.json").read_text())
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2))
    print(f"[soup] averaged {n_avg} float tensors at alpha={args.alpha} -> {out}")
    print(f"[soup] decode: {gen}")


if __name__ == "__main__":
    main()
