#!/usr/bin/env python3
"""Concatenate jsonl corpora with per-file row caps, dedup, shuffle."""
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", action="append", required=True,
                    help="path[:max_rows]; max_rows 0 or absent = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, seen = [], set()
    for spec in args.input:
        path, _, cap = spec.partition(":")
        cap = int(cap) if cap else 0
        got = [json.loads(l) for l in open(path)]
        rng.shuffle(got)
        if cap:
            got = got[:cap]
        n_new = 0
        for r in got:
            k = (r["question"], r["completion"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
            n_new += 1
        print(f"{path}: {len(got)} read, {n_new} kept")
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter(r.get("source") for r in rows))
    print(f"wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
