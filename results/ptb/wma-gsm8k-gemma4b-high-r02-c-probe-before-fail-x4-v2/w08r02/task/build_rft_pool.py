#!/usr/bin/env python3
"""Build the problem pool that rft_sample.py samples solutions for.

Problems only -- no reference solutions. Sources are the GSM8K *train* split
(exact gold answers) and OpenMathInstruct-2's augmented_gsm8k problems, which
are augmentations of the same train split.
"""
from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

from datasets import load_dataset  # noqa: E402

from build_data import MATH_PROMPT_TEMPLATE, fewshot_prefix, norm_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rft_pool.jsonl")
    ap.add_argument("--n-augmented", type=int, default=18000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prefix = fewshot_prefix()
    rows = []

    g = load_dataset("openai/gsm8k", "main", split="train")
    for r in g:
        ans = norm_answer(r["answer"].split("####")[-1])
        if ans is None:
            continue
        rows.append({"question": r["question"].strip(), "answer": ans, "src": "gsm8k_train"})
    print(f"gsm8k train: {len(rows)}")

    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    omi = omi.filter(lambda r: r["problem_source"] == "augmented_gsm8k", num_proc=8)
    idx = list(range(len(omi)))
    rng.shuffle(idx)
    seen = set()
    n0 = len(rows)
    for i in idx:
        if len(rows) - n0 >= args.n_augmented:
            break
        r = omi[i]
        ans = norm_answer(r["expected_answer"])
        q = r["problem"].strip()
        if ans is None or q in seen:
            continue
        seen.add(q)
        rows.append({"question": q, "answer": ans, "src": "augmented_gsm8k"})
    print(f"augmented_gsm8k: {len(rows)-n0}")

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for k, r in enumerate(rows):
            f.write(
                json.dumps(
                    {
                        "id": f"pool-{k}",
                        "user": MATH_PROMPT_TEMPLATE.format(prompt=r["question"]),
                        "answer": r["answer"],
                        "src": r["src"],
                        "system_prefix": prefix,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} problems to {args.out}")


if __name__ == "__main__":
    main()
