#!/usr/bin/env python3
"""Select RFT rows and mix them with a slice of the original SFT file.

Selection favours problems the model finds hard-but-solvable: a problem the
model already solves on every sample carries little gradient, one it never
solves contributes nothing (no correct sample exists), so the rows worth
keeping most are the ones with an intermediate pass rate.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", required=True)
    ap.add_argument("--sft", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-rows", type=int, default=40000)
    ap.add_argument("--keep-easy", type=int, default=1, help="rows kept when pass_rate == 1")
    ap.add_argument("--keep-hard", type=int, default=2, help="rows kept when pass_rate < 1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_q = defaultdict(list)
    for line in open(args.rft):
        r = json.loads(line)
        by_q[r["qid"]].append(r)

    out = []
    n_easy = n_hard = 0
    for qid, rows in by_q.items():
        pr = rows[0].get("pass_rate", 1.0)
        k = args.keep_easy if pr >= 1.0 else args.keep_hard
        rng.shuffle(rows)
        out.extend(rows[:k])
        if pr >= 1.0:
            n_easy += 1
        else:
            n_hard += 1
    print(f"rft problems: {len(by_q)} (all-correct {n_easy}, partial {n_hard}); "
          f"rft rows kept {len(out)}")

    if args.sft and args.sft_rows:
        sft = [json.loads(l) for l in open(args.sft)]
        rng.shuffle(sft)
        out.extend(sft[: args.sft_rows])
        print(f"sft rows mixed in: {min(args.sft_rows, len(sft))}")

    rng.shuffle(out)
    with open(args.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("wrote", args.out, len(out))


if __name__ == "__main__":
    main()
