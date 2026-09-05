#!/usr/bin/env python3
"""Collect the problems a rejection-sampling round should be run on.

Two sources, both GSM8K *train*-side:
  * the openai/gsm8k train split itself (7473 problems with reference answers)
  * OpenMathInstruct-2 shards 8-10, gsm8k-derived rows whose problem text does
    NOT already appear in the SFT corpus the parent was trained on

Writes {question, answer} jsonl. No test item is touched.
"""
from __future__ import annotations

import argparse
import json
import re

import pyarrow.parquet as pq

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data/train-{i:05d}-of-00032.parquet"
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def norm(s):
    s = (s or "").strip().replace(",", "").replace("$", "")
    if not NUM.fullmatch(s):
        return None
    f = float(s)
    return str(int(f)) if f == int(f) else str(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", default="/home/ben/task/data/sft_v3.jsonl")
    ap.add_argument("--shards", default="8,9,10")
    ap.add_argument("--n-new", type=int, default=24000)
    ap.add_argument("--n-seen", type=int, default=25000, help="problems the parent already trained on, resampled for on-policy paths (the classic RFT setup)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # the prompt text of every problem the parent already trained on, with its answer
    seen = {}
    for line in open(args.seen):
        r = json.loads(line)
        body = r["prompt"].split('the answer to the problem.\n\n', 1)[-1].split('\n\nRemember to put', 1)[0]
        if r.get("answer"):
            seen[body.strip()] = r["answer"]

    rows = []
    from datasets import load_dataset
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        a = norm(r["answer"].rsplit("####", 1)[1])
        if a is not None:
            rows.append({"question": r["question"], "answer": a, "src": "gsm8k_train"})
    n_gsm = len(rows)

    added, dedup = 0, set()
    for i in [int(x) for x in args.shards.split(",")]:
        pf = pq.ParquetFile(OMI2.format(i=i))
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=["problem", "expected_answer", "problem_source"]).to_pylist():
                if added >= args.n_new:
                    break
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                q = (r["problem"] or "").strip()
                a = norm(r["expected_answer"])
                if a is None or q in seen or q in dedup:
                    continue
                dedup.add(q)
                rows.append({"question": q, "answer": a, "src": "omi2_unseen"})
                added += 1

    import random
    rng = random.Random(args.seed)
    pool_seen = [q for q in seen if q not in dedup]
    rng.shuffle(pool_seen)
    n_seen = 0
    for q in pool_seen[: args.n_seen]:
        rows.append({"question": q, "answer": seen[q], "src": "omi2_seen"})
        n_seen += 1

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"gsm8k_train": n_gsm, "omi2_unseen": added, "omi2_seen": n_seen, "total": len(rows),
                      "seen_problems_excluded": len(seen)}, indent=2))


if __name__ == "__main__":
    main()
