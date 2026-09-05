#!/usr/bin/env python3
"""Cap / subsample the RFT pool and emit the stage-2 training mix + a decon copy."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def load(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="data/rft1.jsonl")
    ap.add_argument("--out", default="data/mix2.jsonl")
    ap.add_argument("--cap-gsm", type=int, default=3)
    ap.add_argument("--cap-aug", type=int, default=2)
    ap.add_argument("--max-aug", type=int, default=36000)
    ap.add_argument("--metamath", type=int, default=8000)
    args = ap.parse_args()

    rng = random.Random(0)
    gsm = load("data/gsm8k_train.jsonl")
    gsm_qs = {r["question"] for r in gsm}

    by_q = defaultdict(list)
    for r in load(args.rft):
        by_q[r["question"]].append(r)

    rft_gsm, rft_aug = [], []
    for q, rs in by_q.items():
        cap = args.cap_gsm if q in gsm_qs else args.cap_aug
        keep = rs if len(rs) <= cap else rng.sample(rs, cap)
        (rft_gsm if q in gsm_qs else rft_aug).extend(keep)
    rng.shuffle(rft_aug)
    rft_aug = rft_aug[:args.max_aug]

    mm = load("data/metamath_gsm.jsonl", args.metamath)

    mix = rft_gsm + rft_aug + gsm + mm
    rng.shuffle(mix)
    with open(args.out, "w") as f:
        for r in mix:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "_decon.jsonl"), "w") as f:
        for i, r in enumerate(mix):
            f.write(json.dumps({"id": i, "text": r["question"] + "\n" + r["solution"]
                                + "\nANSWER: " + r["answer"]}) + "\n")
    print(f"rft_gsm={len(rft_gsm)} rft_aug={len(rft_aug)} gsm={len(gsm)} mm={len(mm)} "
          f"total={len(mix)}")


if __name__ == "__main__":
    main()
