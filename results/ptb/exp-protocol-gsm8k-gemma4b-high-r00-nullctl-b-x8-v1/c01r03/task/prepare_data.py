#!/usr/bin/env python3
"""Build the SFT dataset for GSM8K post-training of gemma-3-4b-pt.

Sources (all training-split / synthetic-augmentation only, never GSM8K test):
  * nvidia/OpenMathInstruct-2 (train_1M subset): `gsm8k`, `augmented_gsm8k`,
    plus a slice of `math`/`augmented_math` with integer answers for diversity.
  * openai/gsm8k `train` split (human reference solutions), which also matches
    the exact style of the few-shot prefix that the evaluation harness uses.

Every sample is rendered in the harness' prompt format and ends with the
required `ANSWER: <number>` line.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"

INT_RE = re.compile(r"^-?\d+$")
BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(sol: str):
    """Replace the single \\boxed{...} in a solution with its plain content."""
    m = BOXED_RE.search(sol)
    if m is None:
        return None
    if len(BOXED_RE.findall(sol)) != 1:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(sol) and depth:
        if sol[i] == "{":
            depth += 1
        elif sol[i] == "}":
            depth -= 1
        i += 1
    if depth:
        return None
    inner = sol[start : i - 1]
    return sol[: m.start()] + inner + sol[i:]


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    a = a.replace("\\", "").strip()
    if INT_RE.match(a):
        return str(int(a))
    return None


def omi2_rows(max_per_problem: dict):
    files = sorted(glob.glob(OMI2_GLOB))
    assert files, "OpenMathInstruct-2 parquet files not found"
    per_problem = defaultdict(int)
    out = defaultdict(list)
    for f in files:
        t = pq.read_table(f)
        keep = pc.is_in(t.column("problem_source"), value_set=pa.array(list(max_per_problem)))
        t = t.filter(keep)
        for r in t.to_pylist():
            src = r["problem_source"]
            ans = clean_answer(r["expected_answer"])
            if ans is None:
                continue
            key = (src, r["problem"])
            if per_problem[key] >= max_per_problem[src]:
                continue
            sol = r["generated_solution"]
            if len(sol) > 3000 or len(r["problem"]) > 1500:
                continue
            sol = strip_boxed(sol)
            if sol is None:
                continue
            sol = sol.strip()
            # Drop degenerate / self-referential solutions.
            if "\\boxed" in sol or len(sol) < 40:
                continue
            per_problem[key] += 1
            out[src].append(
                {"question": r["problem"].strip(), "solution": sol, "answer": ans, "source": src}
            )
    return out


def gsm8k_train_rows():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in ds:
        q = r["question"].strip()
        reasoning, target = r["answer"].split("####")
        target = target.strip().replace(",", "")
        rows.append(
            {
                "question": q,
                "solution": reasoning.strip(),
                "answer": target,
                "source": "gsm8k_human",
            }
        )
    return rows


def fewshot_block(rows):
    """Render few-shot examples exactly the way inspect_evals/gsm8k does."""
    return "\n\n".join(
        f"{r['question']}\n\nReasoning:\n{r['solution']}\n\nANSWER: {r['answer']}" for r in rows
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-aug-math", type=int, default=25000)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    buckets = omi2_rows({"gsm8k": 4, "augmented_gsm8k": 2, "math": 2, "augmented_math": 1})
    human = gsm8k_train_rows()

    pool = []
    pool += buckets["gsm8k"]
    pool += buckets["augmented_gsm8k"]
    am = buckets["math"] + buckets["augmented_math"]
    rng.shuffle(am)
    pool += am[: args.max_aug_math]
    pool += human

    for src in ["gsm8k", "augmented_gsm8k", "math", "augmented_math"]:
        print(f"  {src}: {len(buckets[src])}")
    print(f"  gsm8k_human: {len(human)}")
    print(f"pool: {len(pool)}")

    rng.shuffle(pool)

    # Few-shot prefixes are drawn only from the human GSM8K *train* split, i.e.
    # the same distribution the harness samples its 10-shot prefix from.
    n_fs = int(len(pool) * args.fewshot_frac)
    ks = [1, 2, 3, 4, 5, 8, 10]

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, r in enumerate(pool):
            completion = f"{r['solution']}\n\nANSWER: {r['answer']}"
            user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
            system = None
            if i < n_fs:
                k = rng.choice(ks)
                shots = rng.sample(human, k)
                system = fewshot_block(shots)
            f.write(
                json.dumps(
                    {
                        "system": system,
                        "user": user,
                        "completion": completion,
                        "question": r["question"],
                        "answer": r["answer"],
                        "source": r["source"],
                    }
                )
                + "\n"
            )
    print(f"wrote {len(pool)} -> {args.out} ({n_fs} with few-shot prefix)")


if __name__ == "__main__":
    main()
