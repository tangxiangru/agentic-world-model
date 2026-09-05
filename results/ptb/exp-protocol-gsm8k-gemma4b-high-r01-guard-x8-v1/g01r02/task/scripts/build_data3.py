#!/usr/bin/env python3
"""Third OpenMathInstruct-2 slice: every row is a chain the model has not seen.

v1 exhausted the *unique problems* in the gsm8k family (105873 of ~119k), so
v3 is disjoint at the level of the solution text rather than the question:
rows whose exact chain already appears in v1 or v2 are dropped, and each
problem may contribute up to --max-per-problem chains in total across v3.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq

from build_data import (OMI2, load_gsm8k_train, numeric, render_completion,
                        render_prompt, strip_boxed)


def used_chains(paths):
    out = set()
    for p in paths:
        for line in open(p):
            c = json.loads(line)["completion"]
            out.add(hash(c.split("\n\nANSWER: ")[0].strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="+", default=["/home/ben/task/data/sft_v1.jsonl",
                                                     "/home/ben/task/data/sft_v2.jsonl"])
    ap.add_argument("--n-gsm", type=int, default=78000)
    ap.add_argument("--n-math", type=int, default=22000)
    ap.add_argument("--max-per-problem", type=int, default=3)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = used_chains(args.exclude)
    print(f"excluding {len(seen)} chains already trained on", flush=True)
    gsm_train = load_gsm8k_train()

    buckets = defaultdict(list)
    per_problem = defaultdict(int)
    want = {"gsm": args.n_gsm, "math": args.n_math}
    files = sorted(glob.glob(OMI2))
    rng.shuffle(files)
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"])
        for r in t.to_pylist():
            fam = "gsm" if "gsm8k" in r["problem_source"] else "math"
            if len(buckets[fam]) >= want[fam]:
                continue
            q = r["problem"].strip()
            if per_problem[q] >= args.max_per_problem:
                continue
            ans = numeric(r["expected_answer"])
            if ans is None:
                continue
            sol = strip_boxed(r["generated_solution"]).strip()
            if not sol or len(sol) > 4000 or "ANSWER:" in sol or "####" in sol:
                continue
            if hash(sol) in seen:
                continue
            per_problem[q] += 1
            buckets[fam].append({"q": q, "r": sol, "a": ans})
        print(f"{f.split('/')[-1]}: gsm={len(buckets['gsm'])} math={len(buckets['math'])}",
              flush=True)
        if all(len(buckets[k]) >= want[k] for k in want):
            break

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
