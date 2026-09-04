#!/usr/bin/env python3
"""Concatenate jsonl training files with per-file row caps and one shuffle."""
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("spec", nargs="+", help="path[:max_rows] , in order")
    args = ap.parse_args()

    rows, counts = [], {}
    for sp in args.spec:
        path, _, cap = sp.partition(":")
        cap = int(cap) if cap else None
        got = []
        with open(path) as f:
            for line in f:
                if cap is not None and len(got) >= cap:
                    break
                got.append(line.rstrip("\n"))
        counts[path] = len(got)
        rows.extend(got)
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(json.dumps({"out": args.out, "total": len(rows), "per_file": counts}, indent=1))


if __name__ == "__main__":
    main()
