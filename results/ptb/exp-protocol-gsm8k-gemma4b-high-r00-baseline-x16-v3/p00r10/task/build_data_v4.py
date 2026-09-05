#!/usr/bin/env python3
"""SFT data, second iteration: OpenMathInstruct-2 only, no raw GSM8K-train solutions.

exp-02 showed the raw human GSM8K-train solutions teach a terse '<<a*b=c>>' style that
the model reaches for whenever a question *looks* like real GSM8K - and that style
scores 0.44 where the verbose style scores 0.67 (memory/cards/exp-02.yaml, conclusion).
So the real GSM8K-train questions stay (they are the benchmark's own distribution) but
they are paired only with OpenMathInstruct-2's verbose, answer-verified solutions.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import fmt  # noqa: E402

OMI2_GLOB = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/"
    "train_*M-*.parquet"
)
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def clean(sol: str) -> str | None:
    if len(BOXED.findall(sol)) != 1:
        return None
    sol = BOXED.sub(r"\1", sol).strip()
    # '<<...>>' is the terse GSM8K calculator annotation exp-02 learned to imitate
    if "\\boxed" in sol or "####" in sol or "ANSWER:" in sol or "<<" in sol:
        return None
    return sol


def is_int(a) -> bool:
    return bool(re.fullmatch(r"-?\d+", str(a).strip().replace(",", "")))


def collect(per_problem_cap: dict[str, int], row_cap: dict[str, int]):
    import pyarrow.parquet as pq

    by_source = {k: [] for k in per_problem_cap}
    per_problem = defaultdict(int)
    seen_bodies = set()
    for path in sorted(glob.glob(OMI2_GLOB)):
        df = pq.read_table(path).to_pandas()
        df = df[df["problem_source"].isin(per_problem_cap)]
        for problem, sol, ans, src in zip(
            df["problem"], df["generated_solution"], df["expected_answer"], df["problem_source"]
        ):
            if len(by_source[src]) >= row_cap[src] or not is_int(ans):
                continue
            problem = problem.strip()
            if per_problem[problem] >= per_problem_cap[src]:
                continue
            body = clean(sol)
            if body is None or not (30 <= len(body) <= 3000):
                continue
            key = (problem, body[:160])
            if key in seen_bodies:
                continue
            seen_bodies.add(key)
            per_problem[problem] += 1
            by_source[src].append((problem, body, str(ans).strip().replace(",", "")))
        print(path.split("/")[-1], {k: len(v) for k, v in by_source.items()}, flush=True)
        if all(len(v) >= row_cap[k] for k, v in by_source.items()):
            break
    return by_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v4.jsonl")
    ap.add_argument("--gsm8k-rows", type=int, default=45000)
    ap.add_argument("--gsm8k-per-problem", type=int, default=8)
    ap.add_argument("--aug-rows", type=int, default=90000)
    ap.add_argument("--aug-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.09)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    fmt.assert_prompt_template_matches()
    rng = random.Random(args.seed)

    by_source = collect(
        {"gsm8k": args.gsm8k_per_problem, "augmented_gsm8k": args.aug_per_problem},
        {"gsm8k": args.gsm8k_rows, "augmented_gsm8k": args.aug_rows},
    )
    records = by_source["gsm8k"] + by_source["augmented_gsm8k"]
    print({k: len(v) for k, v in by_source.items()})
    rng.shuffle(records)

    n_fewshot = 0
    with open(args.out, "w") as f:
        for question, body, answer in records:
            fewshot = rng.random() < args.fewshot_frac
            n_fewshot += fewshot
            rec = {
                "prompt": fmt.render_prompt(question, fewshot=fewshot),
                "completion": fmt.render_target(body, answer),
                "question": question,
                "answer": answer,
                "fewshot": fewshot,
            }
            assert rec["completion"].count(fmt.ANSWER_MARKER) == 1
            assert rec["completion"].rstrip("\n").endswith(fmt.STOP_TOKEN)
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {args.out}: {len(records)} rows, {n_fewshot} with the 10-shot prefix")


if __name__ == "__main__":
    main()
