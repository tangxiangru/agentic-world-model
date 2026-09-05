#!/usr/bin/env python3
"""Mix several JSONL SFT shards into one shuffled training file."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True,
                    help="path:count:repeat entries (count=0 -> all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []
    for s in args.spec:
        parts = s.split(":")
        path = parts[0]
        count = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        repeat = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        rows = [json.loads(l) for l in open(path)]
        rng.shuffle(rows)
        if count:
            rows = rows[:count]
        for _ in range(repeat):
            out.extend(rows)
        print(f"{path}: {len(rows)} x{repeat}")
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(Counter(r.get("source", "?") for r in out))
    print("total", len(out), "->", args.out)


if __name__ == "__main__":
    main()
