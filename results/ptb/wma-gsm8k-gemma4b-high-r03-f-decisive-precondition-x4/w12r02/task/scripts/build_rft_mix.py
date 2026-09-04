#!/usr/bin/env python3
"""Second-stage training file: the model's own verified-correct samples, plus
teacher solutions for exactly the problems it could not solve.

Component A (on-policy): rows written by scripts/rft_sample.py -- up to 2 of the
shortest self-samples per problem whose last number matches gold.
Component B (hard problems): for every problem in the RFT problem pool that
produced no usable correct sample, up to `--teacher-per-problem` OpenMathInstruct-2
solutions the exp-02 run never saw (data/sft_v2_moresols.jsonl).
"""
from __future__ import annotations

import argparse
import json
import random

MARK_A = "the answer to the problem.\n\n"
MARK_B = "\n\nRemember to put your answer"


def qof(prompt: str) -> str:
    a = prompt.find(MARK_A) + len(MARK_A)
    b = prompt.find(MARK_B)
    return prompt[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="data/rft_raw.jsonl")
    ap.add_argument("--problems", default="data/rft_problems.jsonl")
    ap.add_argument("--teacher", default="data/sft_v2_moresols.jsonl")
    ap.add_argument("--out", default="data/rft_mix.jsonl")
    ap.add_argument("--teacher-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rft = [json.loads(l) for l in open(args.rft)]
    solved = {qof(r["prompt"]) for r in rft}
    pool = [json.loads(l) for l in open(args.problems)]
    hard = [p["question"] for p in pool if p["question"] not in solved]
    print(f"[mix] rft rows {len(rft)}, solved problems {len(solved)}, "
          f"unsolved problems {len(hard)} of {len(pool)}")

    hardset = set(hard)
    per = {}
    teacher_rows = []
    with open(args.teacher) as f:
        for line in f:
            r = json.loads(line)
            q = qof(r["prompt"])
            if q not in hardset:
                continue
            if per.get(q, 0) >= args.teacher_per_problem:
                continue
            per[q] = per.get(q, 0) + 1
            r["source"] = "teacher:hard"
            teacher_rows.append(r)
    print(f"[mix] teacher rows {len(teacher_rows)} covering {len(per)} of {len(hard)} "
          f"unsolved problems")

    rows = rft + teacher_rows
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[mix] wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
