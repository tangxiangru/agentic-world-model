#!/usr/bin/env python3
"""Uniform weight average ("model soup") of two or more full fine-tunes of the
same base checkpoint. Config/tokenizer/processor are copied from the first
source; generation_config is written greedy.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def index_of(d: Path) -> dict[str, str]:
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    return idx["weight_map"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    srcs = [Path(s) for s in args.src]
    w = args.weights or [1.0 / len(srcs)] * len(srcs)
    assert len(w) == len(srcs)
    tot = sum(w)
    w = [x / tot for x in w]
    print("averaging", [str(s) for s in srcs], "weights", w)

    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for f in srcs[0].iterdir():
        if f.is_file() and f.suffix not in {".safetensors", ".bin", ".pt"}:
            shutil.copy2(f, dst / f.name)

    wm = index_of(srcs[0])
    shards = sorted(set(wm.values()))
    for shard in shards:
        acc: dict[str, torch.Tensor] = {}
        for src, coef in zip(srcs, w):
            sd = load_file(str(src / shard))
            for k, v in sd.items():
                t = v.to(torch.float32) * coef
                acc[k] = t if k not in acc else acc[k] + t
            del sd
        out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, str(dst / shard), metadata={"format": "pt"})
        print("wrote", shard, len(out), "tensors", flush=True)
        del acc, out

    gc = dst / "generation_config.json"
    g = json.loads(gc.read_text())
    g.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1})
    gc.write_text(json.dumps(g, indent=2) + "\n")
    print("done ->", dst)


if __name__ == "__main__":
    main()
