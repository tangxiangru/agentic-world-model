#!/usr/bin/env python3
"""Teacher chains for the questions the incumbent could not solve.

The rejection-sampling run left a residue: questions where none of 4 samples
ended on the right number. Those are the frontier, and rejection sampling by
construction contributes nothing for them. This pulls every available
OpenMathInstruct-2 solution for exactly those questions out of the train_5M
split, formatted the same way as every other target.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
from build_data import END_OF_TURN, clean_solution, norm_answer, norm_problem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="questions that were sampled")
    ap.add_argument("--solved", required=True, help="rft output: questions with >=1 correct sample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-problem", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = {norm_problem(json.loads(l)["question"]) for l in open(args.pool)}
    solved = {norm_problem(json.loads(l)["question"]) for l in open(args.solved)}
    unsolved = pool - solved
    print(f"pool={len(pool)} solved={len(solved)} unsolved={len(unsolved)}")

    import pyarrow.parquet as pq

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_5M-*.parquet"))
    by_problem: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        c = {k: t.column(k).to_pylist() for k in t.column_names}
        for prob, sol, ans, src in zip(c["problem"], c["generated_solution"],
                                       c["expected_answer"], c["problem_source"]):
            if "gsm8k" not in src:
                continue
            key = norm_problem(prob)
            if key not in unsolved:
                continue
            a = norm_answer(ans)
            s = clean_solution(sol)
            if a is None or s is None or not (80 <= len(s) <= 2600):
                continue
            by_problem[key].append((prob, s, a))

    rng = random.Random(args.seed)
    rows = []
    for key, cands in by_problem.items():
        rng.shuffle(cands)
        seen = set()
        for prob, s, a in cands:
            if s in seen:
                continue
            seen.add(s)
            rows.append({"question": prob, "target": s + "\n\nANSWER: " + a + END_OF_TURN, "answer": a})
            if len(seen) >= args.per_problem:
                break
    rng.shuffle(rows)

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "") + ".decon.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    print(f"unsolved questions with teacher chains={len(by_problem)} rows={len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
