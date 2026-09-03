#!/usr/bin/env python3
"""Round-2 SFT data: more unique GSM8K-style problems, and a few-shot slice
whose prefixes are as long as the one the grader actually presents.

Differences from build_data.py (exp-02):
  * 16 OpenMathInstruct-2 shards instead of 4  -> more unique problems
  * --fewshot-frac 0.25 and k in 4..10 (the grader always sends 10 demos)
  * demos are rendered exactly like inspect_evals' sample_to_fewshot()

Sources are gsm8k / augmented_gsm8k rows, i.e. derived from the GSM8K *train*
split. Nothing here touches the test split.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

from build_data import MATH_PROMPT_TEMPLATE, OMI2, STOP_TOKEN, clean_answer, strip_boxed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_train2.jsonl")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen_problem: set[str] = set()
    rows: list[dict] = []

    for s in range(args.shards):
        path = OMI2 / f"train-{s:05d}-of-00032.parquet"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = clean_answer(r["expected_answer"] or "")
                if ans is None:
                    continue
                prob = (r["problem"] or "").strip()
                if not (20 <= len(prob) <= 1200) or prob in seen_problem:
                    continue
                sol = strip_boxed(r["generated_solution"] or "")
                if not (40 <= len(sol) <= 2500):
                    continue
                if "\\" in sol or "```" in sol:
                    continue
                seen_problem.add(prob)
                rows.append(
                    {
                        "question": prob,
                        "body": sol.rstrip(),
                        "answer": ans,
                    }
                )
        print(f"shard {s}: {len(rows)} unique problems", file=sys.stderr)

    rng.shuffle(rows)
    rows = rows[: args.n]

    n_fs = int(len(rows) * args.fewshot_frac)
    pool = rows[n_fs:]
    out = []
    for i, r in enumerate(rows):
        system = None
        if i < n_fs and len(pool) > args.kmax:
            k = rng.randint(args.kmin, args.kmax)
            demos = rng.sample(pool, k)
            system = "\n\n".join(
                f"{d['question']}\n\nReasoning:\n{d['body']}\n\nANSWER: {d['answer']}"
                for d in demos
            )
        out.append(
            {
                "system": system,
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=r["question"]),
                "completion": f"{r['body']}\n\nANSWER: {r['answer']}{STOP_TOKEN}",
                "answer": r["answer"],
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} rows to {args.out} ({n_fs} few-shot)", file=sys.stderr)


if __name__ == "__main__":
    main()
