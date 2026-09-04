#!/usr/bin/env python3
"""Round-2 SFT data: additional distinct solutions for problems already seen in
round 1 (the fresh-problem pool is exhausted), taken from the full
OpenMathInstruct-2 train split.

Excludes the exact solution bodies already used in round 1 and every problem in
the held-out dev set.
"""
from __future__ import annotations

import argparse
import hashlib
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


def h(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--round1", default="data/sft_omi2_gsm8k.jsonl")
    ap.add_argument("--dev", default="data/dev_fresh_5014.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n", type=int, default=110000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    keep_problems, used_sol = set(), set()
    with open(args.round1) as f:
        for line in f:
            r = json.loads(line)
            keep_problems.add(r["question"])
            used_sol.add(h(r["answer"].strip()))
    dev_problems = set()
    with open(args.dev) as f:
        for line in f:
            dev_problems.add(json.loads(line)["question"])
    print(f"{len(keep_problems)} round-1 problems, {len(used_sol)} used solutions, "
          f"{len(dev_problems)} dev problems held out", flush=True)

    train = load_dataset("openai/gsm8k", "main", split="train")
    demos = []
    for r in train:
        parts = r["answer"].split("####")
        demos.append((r["question"], "####".join(parts[:-1]).strip(), parts[-1].strip()))

    rows: dict[str, list[str]] = {}
    n_rows = 0
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
                if prob in dev_problems or prob not in keep_problems:
                    continue
                bucket = rows.setdefault(prob, [])
                if len(bucket) >= args.max_per_problem:
                    continue
                a = clean_answer(ans)
                if a is None:
                    continue
                c = build_completion(sol, a)
                if c is None:
                    continue
                body = c[: -len(STOP)].strip()
                if h(body) in used_sol:
                    continue
                used_sol.add(h(body))
                bucket.append(c)
                n_rows += 1
        print(f"shard {i}: {n_rows} new solutions over {len(rows)} problems", flush=True)
        if n_rows >= args.n * 1.15:
            break

    flat = [(prob, c) for prob, cs in rows.items() for c in cs]
    rng.shuffle(flat)
    flat = flat[: args.n]
    print(f"writing {len(flat)} rows")

    with open(args.out, "w") as f:
        for prob, comp in flat:
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(2, 10)
                system = "\n\n".join(sample_to_fewshot(*d) for d in rng.sample(demos, k))
            f.write(
                json.dumps(
                    {
                        "prompt": render(system, MATH_PROMPT_TEMPLATE.format(prompt=prob)),
                        "completion": comp,
                        "question": prob,
                        "answer": comp[: -len(STOP)],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
