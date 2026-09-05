#!/usr/bin/env python3
"""Second slice of OpenMathInstruct-2, disjoint in *problems* from data/sft_v1.jsonl.

Shards are walked in reverse so the rows seen first are ones build_data.py never
reached, and any question already used in v1 is skipped outright.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq

from build_data import (GSM8K_TRAIN, OMI2, load_gsm8k_train, numeric,  # noqa: F401
                        render_completion, render_prompt, strip_boxed)

HDR = ('Solve the following math problem step by step. The last line of your response '
       'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
       'answer to the problem.\n\n')
TAIL = '\n\nRemember to put your answer on its own line at the end'


def used_questions(path):
    out = set()
    for line in open(path):
        p = json.loads(line)["prompt"]
        i = p.rindex(HDR) + len(HDR)
        out.add(p[i:p.index(TAIL, i)].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--n-gsm", type=int, default=80000)
    ap.add_argument("--n-math", type=int, default=20000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen_q = used_questions(args.exclude)
    print(f"excluding {len(seen_q)} questions already in v1", flush=True)
    gsm_train = load_gsm8k_train()

    buckets = defaultdict(list)
    per_problem = defaultdict(int)
    want = {"gsm": args.n_gsm, "math": args.n_math}
    for f in sorted(glob.glob(OMI2), reverse=True):
        t = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"])
        for r in t.to_pylist():
            fam = "gsm" if "gsm8k" in r["problem_source"] else "math"
            if len(buckets[fam]) >= want[fam] * 1.3:
                continue
            q = r["problem"].strip()
            if q in seen_q or per_problem[q] >= args.max_per_problem:
                continue
            ans = numeric(r["expected_answer"])
            if ans is None:
                continue
            sol = strip_boxed(r["generated_solution"]).strip()
            if not sol or len(sol) > 4000 or "ANSWER:" in sol or "####" in sol:
                continue
            per_problem[q] += 1
            buckets[fam].append({"q": q, "r": sol, "a": ans})
        print(f"{f.split('/')[-1]}: gsm={len(buckets['gsm'])} math={len(buckets['math'])}",
              flush=True)
        if all(len(buckets[k]) >= want[k] * 1.3 for k in want):
            break

    rng.shuffle(buckets["gsm"])
    rng.shuffle(buckets["math"])
    rows = buckets["gsm"][:args.n_gsm] + buckets["math"][:args.n_math]
    rng.shuffle(rows)
    n_fs = 0
    with open(args.out, "w") as fh:
        for row in rows:
            shots = []
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(gsm_train, rng.randint(1, 4))
                n_fs += 1
            fh.write(json.dumps({
                "prompt": render_prompt(row["q"], shots),
                "completion": render_completion(row["r"], row["a"]),
                "answer": row["a"],
            }) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows ({n_fs} few-shot)")


if __name__ == "__main__":
    main()
