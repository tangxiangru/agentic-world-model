"""Build the SFT file from OpenMathInstruct-2's gsm8k-derived rows.

Source rows are GSM8K *train* problems (solution-augmented) and new problems
generated from GSM8K train problems. Nothing here reads the GSM8K test split;
../contamination_check.py is run over the output separately.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
GSM8K_TRAIN_GLOB = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_RE = re.compile(r"^-?\d+$")


def unbox(sol: str) -> str:
    """Drop the \\boxed wrapper: the grader reads the last number, and the prompt
    explicitly says not to use \\boxed."""
    return BOXED.sub(lambda m: m.group(1), sol)


def fewshot_system(demos) -> str:
    """The shape inspect_evals uses: question, 'Reasoning:', the gsm8k rationale,
    then 'ANSWER: <target>', blocks joined by a blank line."""
    parts = []
    for q, a in demos:
        reasoning, target = a.split("####")
        parts.append(f"{q}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=120000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-sol-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-glob", default=OMI2_GLOB)
    args = ap.parse_args()

    assert render.template_hash() == render.TEMPLATE_SHA256, "gemma3.jinja changed"
    rng = random.Random(args.seed)

    gsm_train = pq.read_table(sorted(glob.glob(GSM8K_TRAIN_GLOB))[0]).to_pylist()
    demo_pool = [(r["question"], r["answer"]) for r in gsm_train]

    rows, per_problem, seen = [], {}, set()
    for f in sorted(glob.glob(args.shard_glob)):
        tbl = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        for r in tbl.to_pylist():
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = r["expected_answer"].strip()
            if not INT_RE.match(ans):
                continue
            sol = r["generated_solution"].strip()
            if len(sol) > args.max_sol_chars or "\\boxed" not in sol:
                continue
            sol = unbox(sol).strip()
            # a second answer marker inside the body would give the grader a
            # competing "ANSWER:" line (pitfall double_answer_format)
            if "\\boxed" in sol or "####" in sol or render.ANSWER_MARKER in sol:
                continue
            key = (r["problem"], sol)
            if key in seen:
                continue
            seen.add(key)
            n = per_problem.get(r["problem"], 0)
            if n >= args.max_per_problem:
                continue
            per_problem[r["problem"]] = n + 1
            rows.append({"problem": r["problem"], "solution": sol, "answer": ans,
                         "source": r["problem_source"]})

    rng.shuffle(rows)
    rows = rows[: args.max_rows]
    print(f"kept {len(rows)} rows over {len(per_problem)} distinct problems", flush=True)

    with open(args.out, "w") as fh:
        for i, r in enumerate(rows):
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.choice([1, 2, 3])
                system = fewshot_system(rng.sample(demo_pool, k))
            fh.write(json.dumps({
                "prompt": render.render_prompt(r["problem"], system=system),
                "completion": render.build_completion(r["solution"], r["answer"]),
                "answer": render.format_answer(r["answer"]),
                "source": r["source"],
                "nshot": 0 if system is None else k,
            }, ensure_ascii=False) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
