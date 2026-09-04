#!/usr/bin/env python3
"""Concatenate jsonl SFT shards (optionally subsampling each) into one file."""
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="path[:n] - n rows sampled without replacement")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for spec in args.input:
        if ":" in spec and not spec.endswith(".jsonl"):
            path, n = spec.rsplit(":", 1)
            n = int(n)
        else:
            path, n = spec, None
        rs = [json.loads(l) for l in open(path)]
        if n is not None and n < len(rs):
            rs = rng.sample(rs, n)
        print(f"{path}: {len(rs)} rows")
        rows.extend(rs)
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    doc = args.out.replace(".jsonl", "_docs.jsonl")
    with open(doc, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    print("total", len(rows), "->", args.out, doc)


if __name__ == "__main__":
    main()
