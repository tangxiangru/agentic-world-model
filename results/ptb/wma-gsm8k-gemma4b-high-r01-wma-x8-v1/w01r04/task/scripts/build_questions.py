#!/usr/bin/env python3
"""Emit {question, gold} for every distinct GSM8K-family training problem in the
pool exp-02 drew from, tagged with whether that problem already appeared in
sft_v1.jsonl. Used as the prompt pool for rejection sampling."""
import argparse
import glob
import json
import random

import pandas as pd

from build_data import CALC_RE, OMI2_GLOB, norm_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--out", default="/home/ben/task/data/questions_pool.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    used = set()
    for line in open(args.sft):
        p = json.loads(line)["prompt"]
        i = p.rfind("Solve the following math problem step by step.")
        used.add(p[i:].split("\n\n")[1].strip())

    from datasets import load_dataset

    rows, seen = [], set()
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        q = r["question"].strip()
        a = norm_answer(r["answer"].split("####")[-1])
        if a is None or q in seen:
            continue
        seen.add(q)
        rows.append({"question": q, "gold": a, "src": "gsm8k_train",
                     "in_sft_v1": q in used})

    for f in sorted(glob.glob(OMI2_GLOB)):
        df = pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
        df = df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
        for p, a, s in zip(df.problem, df.expected_answer, df.problem_source):
            q = p.strip()
            a = norm_answer(str(a))
            if a is None or q in seen:
                continue
            seen.add(q)
            rows.append({"question": q, "gold": a, "src": s, "in_sft_v1": q in used})

    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_new = sum(1 for r in rows if not r["in_sft_v1"])
    print(f"wrote {len(rows)} distinct questions -> {args.out}  "
          f"({n_new} not used in sft_v1)")


if __name__ == "__main__":
    main()
