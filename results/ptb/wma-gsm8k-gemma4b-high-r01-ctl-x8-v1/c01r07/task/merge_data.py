#!/usr/bin/env python3
"""Merge SFT jsonl shards into one training file.

Usage: python merge_data.py --out data/mix.jsonl file.jsonl[:start:end][:xN] ...
  :start:end  take that line slice of the file
  :xN         repeat the slice N times
The merged file is reshuffled with --seed and keeps the prompt/completion schema
build_data.py writes, so train_sft.py reads it unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import random


def parse_spec(spec: str):
    parts = spec.split(":")
    path = parts[0]
    start = end = None
    rep = 1
    for p in parts[1:]:
        if p.startswith("x"):
            rep = int(p[1:])
        elif start is None:
            start = int(p) if p else 0
        else:
            end = int(p) if p else None
    return path, (start or 0), end, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("specs", nargs="+")
    args = ap.parse_args()

    rows = []
    for spec in args.specs:
        path, start, end, rep = parse_spec(spec)
        with open(path) as f:
            lines = f.readlines()
        chunk = lines[start:end]
        rows.extend(chunk * rep)
        print(f"{path}[{start}:{end}] x{rep} -> {len(chunk) * rep} rows")

    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        f.writelines(rows)

    src = collections.Counter(json.loads(l).get("source") for l in rows)
    fs = sum(json.loads(l).get("fewshot", False) for l in rows)
    print(f"wrote {len(rows)} rows to {args.out}; 10-shot rows {fs} "
          f"({fs / max(1, len(rows)):.1%})")
    print("sources:", dict(src))


if __name__ == "__main__":
    main()
