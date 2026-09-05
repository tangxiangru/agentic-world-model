#!/usr/bin/env python3
"""Pull GSM8K-derived rows from OpenMathInstruct-2's full 14M train split whose
*problem* does not already appear in the corpora used so far.

train_1M is exhausted for this task (sft_v1 + sft_v2 hold every GSM8K-derived
row it has), so further coverage has to come from the full split.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from build_data import MATH_PROMPT_TEMPLATE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-glob", default="/home/ben/hf_cache/datasets/nvidia___open_math_instruct-2/default/*/*/open_math_instruct-2-train-000*.arrow")
    ap.add_argument("--used", nargs="*", default=["/home/ben/task/data/sft_v1.jsonl", "/home/ben/task/data/sft_v2.jsonl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=400000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    args = ap.parse_args()

    used = set()
    for p in args.used:
        for line in open(p):
            used.add(json.loads(line)["user"])
    print("already-used prompts:", len(used))

    shards = sorted(glob.glob(args.cache_glob))
    print("shards:", len(shards))

    per_problem: Counter = Counter()
    cols = {"problem": [], "generated_solution": [], "expected_answer": [], "problem_source": []}
    n_seen = 0
    for sh in shards:
        with pa.memory_map(sh, "rb") as src:
            tbl = pa.ipc.open_stream(src).read_all()
        src_col = tbl.column("problem_source").to_pylist()
        prob = tbl.column("problem").to_pylist()
        sol = tbl.column("generated_solution").to_pylist()
        exp = tbl.column("expected_answer").to_pylist()
        n_seen += len(prob)
        for i, s in enumerate(src_col):
            if "gsm8k" not in s:
                continue
            q = prob[i]
            if per_problem[q] >= args.max_per_problem:
                continue
            if MATH_PROMPT_TEMPLATE.format(prompt=q.strip()) in used:
                continue
            per_problem[q] += 1
            cols["problem"].append(q)
            cols["generated_solution"].append(sol[i])
            cols["expected_answer"].append(str(exp[i]))
            cols["problem_source"].append(s)
        print(f"  {sh.split('-')[-1]}: seen {n_seen}, kept {len(cols['problem'])}", flush=True)
        if len(cols["problem"]) >= args.max_rows:
            break

    tbl = pa.table(cols)
    pq.write_table(tbl, args.out)
    print("wrote", tbl.num_rows, "rows,", len(per_problem), "distinct problems ->", args.out)


if __name__ == "__main__":
    main()
