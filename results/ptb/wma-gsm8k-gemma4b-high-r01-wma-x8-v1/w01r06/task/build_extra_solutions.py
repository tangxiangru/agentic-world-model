#!/usr/bin/env python3
"""Second pass over OpenMathInstruct-2: solutions for problems already in
data/sft_v2.jsonl, but solutions that were not used there.

sft_v2 kept at most 2 of the 405B-written solutions per problem. The dataset
holds many more. The unique-problem pool is exhausted (shards 6-11 added 749
new problems), so the only 405B data left is extra solutions to the same
problems: different derivations of the same answers, which is a different
thing from another epoch over the same targets.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

from prepare_data import (END_OF_TURN, INT_RE, PROMPT_TEMPLATE, clean_solution,
                          norm_q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--max-per-problem", type=int, default=3,
                    help="extra solutions per problem, on top of sft_v2's 2")
    ap.add_argument("--max-rows", type=int, default=90000)
    ap.add_argument("--out", default="data/sft_extra.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = {norm_q(json.loads(l)["question"]) for l in open("data/dev250.jsonl")}
    seen_pairs: set[tuple[str, str]] = set()
    known_problems: set[str] = set()
    for line in open("data/sft_v2.jsonl"):
        r = json.loads(line)
        q = r["prompt"].split("\n\nRemember to put your answer")[0]
        q = q.split("is the answer to the problem.\n\n", 1)[-1].strip()
        nq = norm_q(q)
        known_problems.add(nq)
        body = r["completion"].split("\n\nANSWER:")[0]
        seen_pairs.add((nq, body[:120]))
    print(f"problems in sft_v2: {len(known_problems)}, "
          f"solutions already used: {len(seen_pairs)}")

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
        "snapshots/*/data/train-*.parquet"))[: args.shards]
    per_problem: dict[str, int] = {}
    rows: list[dict] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "generated_solution",
                                              "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = (r["expected_answer"] or "").strip()
                if not INT_RE.match(ans):
                    continue
                nq = norm_q(r["problem"])
                if nq in dev or nq not in known_problems:
                    continue
                if per_problem.get(nq, 0) >= args.max_per_problem:
                    continue
                sol = clean_solution(r["generated_solution"] or "")
                if sol is None or len(sol) < 20 or "answer:" in sol.lower():
                    continue
                key = (nq, sol[:120])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                per_problem[nq] = per_problem.get(nq, 0) + 1
                rows.append({
                    "prompt": PROMPT_TEMPLATE.format(prompt=r["problem"].strip()),
                    "completion": f"{sol}\n\nANSWER: {ans}{END_OF_TURN}",
                    "answer": ans,
                    "source": "extra:" + r["problem_source"],
                    "system": None,
                })
        print(os.path.basename(path), len(rows), flush=True)
        if len(rows) >= args.max_rows * 2:
            break

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_rows]
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"rows": len(rows), "problems": len(per_problem),
                      "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
