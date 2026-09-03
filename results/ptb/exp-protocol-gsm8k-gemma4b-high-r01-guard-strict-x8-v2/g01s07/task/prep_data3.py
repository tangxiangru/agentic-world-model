#!/usr/bin/env python3
"""Third-pass SFT data: raise the per-problem solution cap and emit only rows
not already used by earlier cards."""
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
    ap.add_argument("--exclude", default="", help="comma-separated jsonl files")
    ap.add_argument("--max-per-problem", type=int, default=8)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--sources", default="gsm8k,augmented_gsm8k")
    ap.add_argument("--shards", default="train_2M")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    seen_sol: set[int] = set()
    prior: dict[str, int] = defaultdict(int)
    for path in [p for p in args.exclude.split(",") if p]:
        with open(path) as fh:
            for line in fh:
                d = json.loads(line)
                seen_sol.add(hash((d["question"], d["answer"][: -len(STOP)])))
                prior[d["question"]] += 1
    print(f"excluding {len(seen_sol)} prior rows over {len(prior)} problems", flush=True)

    sources = set(args.sources.split(","))
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
        f"469216e3f46f4dacf476b382e192485ea51a143e/data/{args.shards}-*.parquet"))
    assert files, "no shards found"

    per: dict[str, list[str]] = defaultdict(list)
    answers: dict[str, str] = {}
    stats: dict[str, int] = defaultdict(int)
    for f in files:
        for rec in pq.read_table(f).to_pylist():
            if rec["problem_source"] not in sources:
                continue
            ans = (rec["expected_answer"] or "").strip()
            if not NUM_RE.match(ans):
                stats["bad_answer"] += 1
                continue
            prob = (rec["problem"] or "").strip()
            if not prob or len(prob) > 1500:
                continue
            tgt = clean_solution(rec["generated_solution"] or "", ans)
            if tgt is None or len(tgt) > 4000:
                continue
            h = hash((prob, tgt))
            if h in seen_sol:
                stats["already_used"] += 1
                continue
            seen_sol.add(h)
            if len(per[prob]) + prior.get(prob, 0) >= args.max_per_problem:
                stats["over_cap"] += 1
                continue
            per[prob].append(tgt)
            answers[prob] = ans
            stats["kept"] += 1

    rows = [{"question": p, "prompt": MATH_PROMPT_TEMPLATE.format(prompt=p),
             "answer": t + STOP, "expected_answer": answers[p]}
            for p, ts in per.items() for t in ts]
    random.Random(args.seed).shuffle(rows)
    if args.n:
        rows = rows[: args.n]
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({**stats, "problems": len(per), "emitted": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
