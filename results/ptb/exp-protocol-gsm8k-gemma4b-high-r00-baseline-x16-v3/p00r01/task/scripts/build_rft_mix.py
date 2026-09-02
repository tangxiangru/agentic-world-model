#!/usr/bin/env python3
"""Mix the rejection-sampled rows with a replay slice of the original SFT corpus.

Replay is not optional: training a checkpoint only on its own correct samples
collapses solution style and forgets the problems it cannot yet solve, which are
exactly the ones the RFT file has no row for.
"""
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rft = [json.loads(l) for l in open(args.rft)]
    sft = [json.loads(l) for l in open(args.sft)]

    rft_qs = {r["question"] for r in rft}
    # prefer replaying problems RFT could NOT solve: those are where the
    # reference solution is the only signal the model has.
    unsolved = [r for r in sft if r["question"] not in rft_qs]
    solved = [r for r in sft if r["question"] in rft_qs]
    rng.shuffle(unsolved)
    rng.shuffle(solved)
    replay = (unsolved + solved)[: args.replay]

    rows = rft + replay
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"rft={len(rft)} replay={len(replay)} "
          f"(unsolved_available={len(unsolved)}) total={len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
