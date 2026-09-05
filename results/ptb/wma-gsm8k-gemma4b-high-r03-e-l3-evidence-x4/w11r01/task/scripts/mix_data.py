#!/usr/bin/env python3
"""Merge jsonl SFT files into one training file, with a replay slice.

Continuing a checkpoint on freshly generated on-policy data alone drifts it off
the corpus it was tuned on; a replay slice of the parent's own training data is
the cheap guard. Each --file is given as path:n (n=0 means the whole file).
"""
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True, help="path:n")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, provenance = [], []
    for spec in args.file:
        path, _, n = spec.rpartition(":")
        n = int(n)
        rs = [json.loads(l) for l in open(path)]
        rng.shuffle(rs)
        if n:
            rs = rs[:n]
        provenance.append((path, len(rs)))
        rows.extend(rs)
    rng.shuffle(rows)

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps({"question": r["question"], "target": r["target"],
                                 "answer": r["answer"]}) + "\n")
    with open(args.out.replace(".jsonl", "") + ".decon.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    print(f"{provenance} -> {len(rows)} rows in {args.out}")


if __name__ == "__main__":
    main()
