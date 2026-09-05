#!/usr/bin/env python3
"""Uniform weight average of two or more checkpoints of the same architecture.

Both parents are fine-tunes of the same base snapshot, so the averaged weights
stay in the same basin. Writes a full, loadable model directory (config,
tokenizer, processor configs, greedy generation_config).
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
    ap.add_argument("--parent", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="comma-separated, defaults to uniform")
    a = ap.parse_args()

    ps = a.parent
    ws = [float(x) for x in a.weights.split(",")] if a.weights else [1 / len(ps)] * len(ps)
    assert len(ws) == len(ps) and abs(sum(ws) - 1) < 1e-6, (ws, ps)
    print("averaging", list(zip(ps, ws)))

    os.makedirs(a.out, exist_ok=True)
    index = json.load(open(os.path.join(ps[0], "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))

    for shard in shards:
        acc = None
        for p, w in zip(ps, ws):
            sd = load_file(os.path.join(p, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) * w for k, v in sd.items()}
            else:
                for k in acc:
                    acc[k] += sd[k].to(torch.float32) * w
            del sd
        out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(a.out, shard), metadata={"format": "pt"})
        print("wrote", shard)
        del acc, out

    for f in ("model.safetensors.index.json", "config.json", "added_tokens.json",
              "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
              "tokenizer_config.json", "preprocessor_config.json",
              "processor_config.json"):
        src = os.path.join(ps[0], f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(a.out, f))

    gc = json.load(open(os.path.join(ps[0], "generation_config.json")))
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    json.dump(gc, open(os.path.join(a.out, "generation_config.json"), "w"), indent=2)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
