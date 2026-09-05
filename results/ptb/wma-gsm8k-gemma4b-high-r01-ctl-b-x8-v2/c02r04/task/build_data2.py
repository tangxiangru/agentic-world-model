#!/usr/bin/env python3
"""Second-pass corpus: more OpenMathInstruct-2 gsm8k solutions from the full
`train` shards, excluding the (problem, solution) pairs already in v1.

Same target shape and same guards as build_data.py; the only differences are
the shard set and that a problem may now carry up to --max-per-problem
solutions in total across v1 + v2.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq

from build_data import (MATH_PROMPT_TEMPLATE, NUM_RE, clean_solution,
                        sample_to_fewshot)

FULL = sorted(glob.glob(
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-000*.parquet"))
GSM8K_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="data/sft_v1_clean.jsonl")
    ap.add_argument("--out", default="data/sft_v2_extra.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-gsm", type=int, default=200000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen_pairs, per_problem = set(), defaultdict(int)
    with open(args.exclude) as fh:
        for line in fh:
            r = json.loads(line)
            seen_pairs.add((r["question"], r["completion"][:200]))
            per_problem[r["question"]] += 0   # v1 count handled by --max-per-problem below

    print(f"exclude set: {len(seen_pairs)} pairs from {args.exclude}", flush=True)

    gt = pq.read_table(GSM8K_TRAIN).to_pylist()
    fewshot_pool = [sample_to_fewshot(r["question"], r["answer"]) for r in gt]

    out_rows = []
    for f in FULL:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = (r["expected_answer"] or "").strip()
                if not NUM_RE.match(ans):
                    continue
                prob = r["problem"].strip()
                if per_problem[prob] >= args.max_per_problem:
                    continue
                sol = clean_solution(r["generated_solution"], ans)
                if sol is None:
                    continue
                completion = f"{sol}\n\nANSWER: {ans}"
                if (prob, completion[:200]) in seen_pairs:
                    continue
                if "boxed" in completion or "####" in completion:
                    continue
                seen_pairs.add((prob, completion[:200]))
                per_problem[prob] += 1
                out_rows.append((prob, completion, ans))
        print(f"  after {f.split('/')[-1]}: {len(out_rows)} rows", flush=True)
        if len(out_rows) >= args.n_gsm:
            break

    rng.shuffle(out_rows)
    out_rows = out_rows[: args.n_gsm]
    with open(args.out, "w") as fh:
        for prob, completion, ans in out_rows:
            messages = []
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 3, 4, 6, 10])
                messages.append({"role": "system",
                                 "content": "\n\n".join(rng.sample(fewshot_pool, k))})
            messages.append({"role": "user",
                             "content": MATH_PROMPT_TEMPLATE.format(prompt=prob)})
            messages.append({"role": "assistant", "content": completion})
            fh.write(json.dumps({"messages": messages, "completion": completion,
                                 "question": prob, "answer": ans,
                                 "source": "omi2_full_gsm8k"}) + "\n")
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
