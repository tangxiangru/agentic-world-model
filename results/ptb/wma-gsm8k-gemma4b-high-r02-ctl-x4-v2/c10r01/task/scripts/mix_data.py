#!/usr/bin/env python3
"""Mix jsonl SFT files with per-file caps and optional filters."""
from __future__ import annotations

import argparse
import json
import random
import re

BACKTRACK = re.compile(r"\bHowever\b|\bWait\b|\bBut wait\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True,
                    help="file:n[:drop_backtrack] triples, n=-1 for all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []
    for spec in args.spec:
        parts = spec.split(":")
        path, n = parts[0], int(parts[1])
        drop_bt = len(parts) > 2 and parts[2] == "drop_backtrack"
        rows = []
        n_bt = 0
        for line in open(path):
            d = json.loads(line)
            if drop_bt and BACKTRACK.search(d["completion"]):
                n_bt += 1
                continue
            rows.append(d)
        rng.shuffle(rows)
        if n >= 0:
            rows = rows[:n]
        print(f"{path}: {len(rows)} rows (dropped {n_bt} backtracking)", flush=True)
        out.extend(rows)
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
