#!/usr/bin/env python3
"""Second-stage pool: GSM8K-derived rows from OpenMathInstruct-2 train_5M that
are not already in the exp-02 corpus.

Same rewriting rules as build_data.py (imported, not duplicated), then every
(problem, target) pair already present in data/sft_pool.jsonl is removed so a
continuation run sees genuinely new solutions rather than a third epoch.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter

import pyarrow.parquet as pq

from build_data import KEEP_SOURCES, last_number, make_target, normalise_answer

GLOB5M = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_5M-*.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_pool2.jsonl")
    ap.add_argument("--exclude", default="/home/ben/task/data/sft_pool.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seen_pairs = set()
    seen_counts = Counter()
    for line in open(args.exclude):
        r = json.loads(line)
        seen_pairs.add((r["problem"], r["target"]))
        seen_counts[r["problem"]] += 1
    print(f"excluding {len(seen_pairs)} (problem, target) pairs already in the exp-02 pool")

    stats = Counter()
    out_rows = []
    per_problem = Counter()
    for path in sorted(glob.glob(GLOB5M)):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(
                rg,
                columns=["problem", "generated_solution", "expected_answer", "problem_source"],
            )
            for r in tbl.to_pylist():
                if r["problem_source"] not in KEEP_SOURCES:
                    continue
                stats["seen"] += 1
                ans = normalise_answer(r["expected_answer"])
                if ans is None:
                    continue
                tgt = make_target(r["generated_solution"], ans)
                if tgt is None:
                    continue
                if last_number(tgt) != last_number("x " + ans):
                    stats["drop_last_number_mismatch"] += 1
                    continue
                n_words = len(tgt.split())
                if n_words < 12 or n_words > 600:
                    continue
                prob = r["problem"].strip()
                if (prob, tgt) in seen_pairs:
                    stats["drop_dup_of_exp02"] += 1
                    continue
                budget = args.max_per_problem - seen_counts[prob]
                if per_problem[prob] >= budget:
                    stats["drop_over_per_problem_cap"] += 1
                    continue
                per_problem[prob] += 1
                seen_pairs.add((prob, tgt))
                out_rows.append(
                    {"problem": prob, "target": tgt, "answer": ans,
                     "source": r["problem_source"], "n_words": n_words}
                )
                stats["kept"] += 1
        print(f"  {path.split('/')[-1]}: kept so far {stats['kept']}", flush=True)

    random.Random(args.seed).shuffle(out_rows)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"stats {dict(stats)}")
    print(f"wrote {len(out_rows)} rows -> {args.out}")
    print(f"by source: {Counter(r['source'] for r in out_rows)}")


if __name__ == "__main__":
    main()
