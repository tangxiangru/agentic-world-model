#!/usr/bin/env python3
"""Select the exp-05 training mix from the raw rejection samples.

Keeps more solutions for problems the exp-04 checkpoint finds hard (low pass rate) and
one for the ones it already solves every time, then mixes in a slice of the original
off-policy corpus so the round does not collapse onto 7.5k problems.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def keep_for(pr: float) -> int:
    if pr <= 0.375:
        return 4
    if pr <= 0.75:
        return 3
    if pr < 1.0:
        return 2
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="/home/ben/task/data/rft_raw.jsonl")
    ap.add_argument("--sft", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--n-sft", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_pid = defaultdict(list)
    for line in open(args.rft):
        r = json.loads(line)
        by_pid[r["pid"]].append(r)

    rows = []
    hist = defaultdict(int)
    for pid, rs in by_pid.items():
        pr = rs[0]["pass_rate"]
        k = keep_for(pr)
        rng.shuffle(rs)
        # prefer the shorter correct solutions: less chance of a rambling chain
        rs.sort(key=lambda r: len(r["target"]))
        for r in rs[:k]:
            rows.append({"question": r["question"], "target": r["target"], "source": "rft"})
        hist[k] += 1
    n_rft = len(rows)

    sft = [json.loads(l) for l in open(args.sft)]
    rng.shuffle(sft)
    rows += sft[: args.n_sft]
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"rft_rows": n_rft, "sft_rows": min(args.n_sft, len(sft)), "total": len(rows),
                      "problems": len(by_pid), "keep_hist": dict(hist)}))


if __name__ == "__main__":
    main()
