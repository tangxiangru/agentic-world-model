#!/usr/bin/env python3
"""Second-stage SFT data: maximise the number of *unique* GSM8K-style problems.

Reads the full OpenMathInstruct-2 train split, keeps one solution per unique
`gsm8k` / `augmented_gsm8k` problem with an integer answer, and drops every
problem already present in the stage-1 file.
"""
from __future__ import annotations

import argparse
import glob
import json
import random

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prepare_data import MATH_PROMPT_TEMPLATE, clean_answer, fewshot_block, gsm8k_train_rows, strip_boxed

FULL_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-000*.parquet"
SOURCES = ["gsm8k", "augmented_gsm8k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft2.jsonl")
    ap.add_argument("--exclude", default="data/sft.jsonl")
    ap.add_argument("--max-problems", type=int, default=170000)
    ap.add_argument("--sols-per-problem", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    used = set()
    if args.exclude:
        for l in open(args.exclude):
            used.add(json.loads(l)["question"])
    print(f"[stage2] {len(used)} problems already used", flush=True)

    best: dict[str, tuple[str, str]] = {}
    counts: dict[str, int] = {}
    files = sorted(glob.glob(FULL_GLOB))
    assert len(files) == 32, len(files)
    for n, f in enumerate(files):
        t = pq.read_table(f)
        t = t.filter(pc.is_in(t.column("problem_source"), value_set=pa.array(SOURCES)))
        for r in t.to_pylist():
            q = r["problem"].strip()
            if q in used or len(q) > 1200:
                continue
            if counts.get(q, 0) >= args.sols_per_problem:
                continue
            a = clean_answer(r["expected_answer"])
            if a is None:
                continue
            sol = r["generated_solution"]
            if len(sol) > 2500:
                continue
            sol = strip_boxed(sol)
            if sol is None:
                continue
            sol = sol.strip()
            if "\\boxed" in sol or len(sol) < 40:
                continue
            counts[q] = counts.get(q, 0) + 1
            best.setdefault(q, (sol, a))
        print(f"  file {n+1}/32 -> {len(best)} unique problems", flush=True)
        if len(best) >= args.max_problems * 1.05:
            break

    items = [(k, v[0], v[1]) for k, v in best.items()]
    rng.shuffle(items)
    items = items[: args.max_problems]

    human = gsm8k_train_rows()
    n_fs = int(len(items) * args.fewshot_frac)
    ks = [1, 2, 3, 4, 5, 8, 10]
    with open(args.out, "w") as f:
        for i, (q, sol, a) in enumerate(items):
            system = fewshot_block(rng.sample(human, rng.choice(ks))) if i < n_fs else None
            f.write(
                json.dumps(
                    {
                        "system": system,
                        "user": MATH_PROMPT_TEMPLATE.format(prompt=q),
                        "completion": f"{sol}\n\nANSWER: {a}",
                        "question": q,
                        "answer": a,
                        "source": "omi2_stage2",
                    }
                )
                + "\n"
            )
    print(f"[stage2] wrote {len(items)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
