#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Sources (all GSM8K-*train*-derived or MATH-train-derived; nothing touches the
GSM8K test split):
  nvidia/OpenMathInstruct-2  problem_source in {gsm8k, augmented_gsm8k, math, augmented_math}

Output: one jsonl with {prompt, completion, question, answer, n_shots, source}.
`question`/`answer` are there so ../contamination_check.py can read the file
directly.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

OMI2 = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data/train-%05d-of-00032.parquet"
)
GSM8K_TRAIN = (
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"
)

EXACT_10SHOT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_10shot_prefix.txt")).read()

NUMERIC = re.compile(r"^-?\d{1,12}(\.\d{1,6})?$")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
DOLLAR_BOXED = re.compile(r"\$\\boxed\{([^{}]*)\}\$")


def unbox(text: str) -> str:
    text = DOLLAR_BOXED.sub(r"\1", text)
    text = BOXED.sub(r"\1", text)
    return text


def clean_solution(sol: str) -> str | None:
    sol = unbox(sol).strip()
    if "\\boxed" in sol or "boxed" in sol:
        return None
    if "ANSWER:" in sol or "ANSWER :" in sol:
        return None
    # OMI-2 occasionally leaves an empty final line
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol or None


def load_fewshot_pool(rng: random.Random, n: int = 4000):
    """Few-shot demonstrations rendered exactly like the grader renders its own
    (question / Reasoning: / ANSWER:), drawn from the GSM8K TRAIN split."""
    t = pq.read_table(GSM8K_TRAIN).to_pylist()
    pool = []
    for r in t[:n]:
        q = r["question"]
        a = r["answer"]
        if "####" not in a:
            continue
        reasoning, target = a.split("####")
        pool.append(fmt.sample_to_fewshot(q, reasoning.strip(), target.strip()))
    rng.shuffle(pool)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--exact-frac", type=float, default=0.0,
                    help="fraction of rows carrying the grader's real fixed 10-shot prefix")
    ap.add_argument("--n-gsm", type=int, default=140000)
    ap.add_argument("--n-math", type=int, default=20000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.30)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="jsonl files whose questions must not reappear")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_fewshot_pool(rng)
    print(f"fewshot pool: {len(pool)}", flush=True)

    seen: dict[str, int] = {}
    for xp in args.exclude:
        with open(xp) as fh:
            for line in fh:
                seen[json.loads(line)["question"].strip()[:200]] = 10**6
    if args.exclude:
        print(f"excluding {len(seen)} questions seen in {args.exclude}", flush=True)
    gsm_rows: list[dict] = []
    math_rows: list[dict] = []

    for i in range(args.shard_start, args.shard_start + args.shards):
        tbl = pq.read_table(OMI2 % i).to_pylist()
        for r in tbl:
            src = r["problem_source"]
            ans = (r["expected_answer"] or "").strip()
            if not NUMERIC.match(ans):
                continue
            bucket = gsm_rows if src in ("gsm8k", "augmented_gsm8k") else math_rows
            if src in ("gsm8k", "augmented_gsm8k"):
                if len(gsm_rows) >= args.n_gsm:
                    continue
            else:
                if len(math_rows) >= args.n_math:
                    continue
            prob = r["problem"].strip()
            key = prob[:200]
            if seen.get(key, 0) >= args.max_per_problem:
                continue
            sol = clean_solution(r["generated_solution"])
            if sol is None or len(sol) < 40:
                continue
            seen[key] = seen.get(key, 0) + 1
            bucket.append({"question": prob, "answer": ans, "solution": sol, "source": src})
        print(f"shard {i}: gsm={len(gsm_rows)} math={len(math_rows)}", flush=True)
        if len(gsm_rows) >= args.n_gsm and len(math_rows) >= args.n_math:
            break

    rows = gsm_rows + math_rows
    rng.shuffle(rows)
    print(f"total {len(rows)} (gsm {len(gsm_rows)}, math {len(math_rows)})", flush=True)

    n_written = 0
    with open(args.out, "w") as f:
        for r in rows:
            k = 0
            u = rng.random()
            if u < args.exact_frac:
                system = EXACT_10SHOT
                k = 10
            elif u < args.exact_frac + args.fewshot_frac:
                k = rng.choice([1, 2, 3])
                system = "\n\n".join(rng.sample(pool, k))
            else:
                system = None
            prompt = fmt.render_prompt(r["question"], system)
            completion = fmt.render_completion(r["solution"], r["answer"])
            assert completion.count(fmt.ANSWER_MARKER) == 1, completion[-200:]
            assert completion.endswith(fmt.STOP_TOKEN)
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "question": r["question"],
                        "answer": r["solution"] + "\n\nANSWER: " + r["answer"],
                        "n_shots": k,
                        "source": r["source"],
                    }
                )
                + "\n"
            )
            n_written += 1
    print(f"wrote {n_written} -> {args.out}")


if __name__ == "__main__":
    main()
