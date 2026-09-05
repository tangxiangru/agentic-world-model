#!/usr/bin/env python3
"""Second-round SFT data: fresh GSM8K-derived problems from the full
OpenMathInstruct-2 train split, excluding every problem already used in round 1.

Same formatting contract as build_data.py (grader template, single 'ANSWER: n'
line, <end_of_turn> terminator).
"""
from __future__ import annotations

import argparse
import json
import random

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from build_data import (
    MATH_PROMPT_TEMPLATE,
    STOP,
    build_completion,
    clean_answer,
    render,
    sample_to_fewshot,
)

REV = "469216e3f46f4dacf476b382e192485ea51a143e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--exclude", default="data/sft_omi2_gsm8k.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    used = set()
    if args.exclude:
        with open(args.exclude) as f:
            for line in f:
                used.add(json.loads(line)["question"])
    print(f"excluding {len(used)} problems already used", flush=True)

    train = load_dataset("openai/gsm8k", "main", split="train")
    demos = []
    for r in train:
        parts = r["answer"].split("####")
        demos.append((r["question"], "####".join(parts[:-1]).strip(), parts[-1].strip()))

    rows: dict[str, list[tuple[str, str]]] = {}
    for i in range(args.shards):
        p = hf_hub_download(
            "nvidia/OpenMathInstruct-2",
            f"data/train-{i:05d}-of-00032.parquet",
            repo_type="dataset",
            revision=REV,
        )
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(
            batch_size=20000,
            columns=["problem", "generated_solution", "expected_answer", "problem_source"],
        ):
            d = batch.to_pydict()
            for prob, sol, ans, src in zip(
                d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
            ):
                if src not in ("gsm8k", "augmented_gsm8k"):
                    continue
                if prob in used:
                    continue
                a = clean_answer(ans)
                if a is None:
                    continue
                bucket = rows.setdefault(prob, [])
                if len(bucket) >= args.max_per_problem:
                    continue
                c = build_completion(sol, a)
                if c is None:
                    continue
                bucket.append((c, a))
        print(f"shard {i}: {len(rows)} fresh problems", flush=True)
        if len(rows) >= args.n * 1.2:
            break

    flat = [(prob, c, a) for prob, sols in rows.items() for (c, a) in sols]
    rng.shuffle(flat)
    flat = flat[: args.n]
    print(f"writing {len(flat)} rows from {len(rows)} fresh problems")

    with open(args.out, "w") as f:
        for prob, comp, a in flat:
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(2, 10)
                system = "\n\n".join(sample_to_fewshot(*p) for p in rng.sample(demos, k))
            prompt = render(system, MATH_PROMPT_TEMPLATE.format(prompt=prob))
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": comp,
                        "question": prob,
                        "answer": comp[: -len(STOP)],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
