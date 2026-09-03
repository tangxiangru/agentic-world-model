#!/usr/bin/env python3
"""Second-pass SFT data: GSM8K-sourced rows from OpenMathInstruct-2 train_2M
that are NOT already in an existing jsonl (exp-02's file).

Same target format as prep_data.py: <reasoning>\n\nANSWER: <n><end_of_turn>.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq

from prep_data import MATH_PROMPT_TEMPLATE, NUM_RE, STOP, clean_solution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--max-per-problem", type=int, default=4)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--sources", default="gsm8k,augmented_gsm8k")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    seen_sol: set[int] = set()
    prior_per_problem: dict[str, int] = defaultdict(int)
    if args.exclude:
        with open(args.exclude) as fh:
            for line in fh:
                d = json.loads(line)
                tgt = d["answer"][: -len(STOP)]
                seen_sol.add(hash((d["question"], tgt)))
                prior_per_problem[d["question"]] += 1
        print(f"excluding {len(seen_sol)} prior rows over "
              f"{len(prior_per_problem)} problems", flush=True)

    sources = set(args.sources.split(","))
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
        "469216e3f46f4dacf476b382e192485ea51a143e/data/train_2M-*.parquet"))
    assert files, "no train_2M shards found"

    per_problem: dict[str, list[str]] = defaultdict(list)
    answers: dict[str, str] = {}
    stats: dict[str, int] = defaultdict(int)

    for f in files:
        for rec in pq.read_table(f).to_pylist():
            stats["seen"] += 1
            if rec["problem_source"] not in sources:
                continue
            stats["src_ok"] += 1
            ans = (rec["expected_answer"] or "").strip()
            if not NUM_RE.match(ans):
                stats["bad_answer"] += 1
                continue
            prob = (rec["problem"] or "").strip()
            if not prob or len(prob) > 1500:
                stats["bad_problem"] += 1
                continue
            tgt = clean_solution(rec["generated_solution"] or "", ans)
            if tgt is None or len(tgt) > 4000:
                stats["bad_solution"] += 1
                continue
            h = hash((prob, tgt))
            if h in seen_sol:
                stats["already_used"] += 1
                continue
            seen_sol.add(h)
            if len(per_problem[prob]) + prior_per_problem.get(prob, 0) >= args.max_per_problem:
                stats["over_cap"] += 1
                continue
            per_problem[prob].append(tgt)
            answers[prob] = ans
            stats["kept"] += 1

    rows = [{"question": p, "prompt": MATH_PROMPT_TEMPLATE.format(prompt=p),
             "answer": t + STOP, "expected_answer": answers[p]}
            for p, ts in per_problem.items() for t in ts]
    random.Random(args.seed).shuffle(rows)
    if args.n:
        rows = rows[: args.n]
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    new_problems = sum(1 for p in per_problem if p not in prior_per_problem)
    print(json.dumps({**stats, "problems_touched": len(per_problem),
                      "problems_new": new_problems, "emitted": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
