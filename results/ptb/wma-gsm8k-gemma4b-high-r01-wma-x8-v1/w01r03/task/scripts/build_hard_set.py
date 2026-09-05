#!/usr/bin/env python3
"""Build a training set concentrated on the problems the current model fails.

Input: a labels jsonl from rft_sample.py ({problem, answer, n_correct, k}) and
one or more OpenMathInstruct-2 corpora.  Output: an SFT jsonl that takes up to
--hard-per-problem teacher solutions for every problem the model got wrong and
up to --easy-per-problem for the rest, so the mixture is weighted toward the
failures without training only on them.
"""
from __future__ import annotations

import argparse
import collections
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hard-per-problem", type=int, default=4)
    ap.add_argument("--easy-per-problem", type=int, default=1)
    ap.add_argument("--easy-share", type=float, default=0.35,
                    help="target share of output rows coming from solved problems")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hard, easy = set(), set()
    with open(args.labels) as f:
        for line in f:
            r = json.loads(line)
            (hard if r["n_correct"] == 0 else easy).add(r["problem"])
    print(f"labels: {len(hard)} unsolved, {len(easy)} solved")

    by_problem: dict[str, list[dict]] = collections.defaultdict(list)
    for path in args.corpus:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["problem"] in hard or r["problem"] in easy:
                    by_problem[r["problem"]].append(r)

    rng = random.Random(args.seed)
    hard_rows, easy_rows = [], []
    for p, rows in by_problem.items():
        rng.shuffle(rows)
        if p in hard:
            hard_rows.extend(rows[: args.hard_per_problem])
        else:
            easy_rows.extend(rows[: args.easy_per_problem])

    n_easy = min(len(easy_rows), int(len(hard_rows) * args.easy_share / max(1e-9, 1 - args.easy_share)))
    rng.shuffle(easy_rows)
    out = hard_rows + easy_rows[:n_easy]
    rng.shuffle(out)

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "hard_problems_with_solutions": sum(1 for p in by_problem if p in hard),
        "hard_rows": len(hard_rows),
        "easy_rows_kept": n_easy,
        "total": len(out),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
