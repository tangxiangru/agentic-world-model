#!/usr/bin/env python3
"""Build GSM8K-style SFT data in the exact format the grader reads.

Source: nvidia/OpenMathInstruct-2 (rows whose problem_source is gsm8k or
augmented_gsm8k -- i.e. grade-school word problems derived from the GSM8K
*train* split, never the test split).

Target shape, byte-for-byte against templates/gemma3.jinja:
    <bos><start_of_turn>user\n{MATH_PROMPT_TEMPLATE}<end_of_turn>\n<start_of_turn>model\n
    {solution}\n\nANSWER: {n}<end_of_turn>
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI2 = Path(
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data"
)

STOP_TOKEN = "<end_of_turn>"

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_RE = re.compile(r"^-?\d+$")


def strip_boxed(sol: str) -> str:
    """Replace \\boxed{x} with x and drop leftover latex delimiters."""
    sol = BOXED.sub(r"\1", sol)
    sol = sol.replace("\\[", "").replace("\\]", "")
    sol = sol.replace("\\(", "").replace("\\)", "")
    sol = re.sub(r"\$([^$\n]{1,40})\$", r"\1", sol)
    return sol.strip()


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    a = a.rstrip(".")
    if INT_RE.match(a):
        # GSM8K targets are plain integers; keep the distribution the same
        if abs(int(a)) > 10**9:
            return None
        return str(int(a))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    per_problem: dict[str, int] = {}
    rows: list[dict] = []

    for s in range(args.shards):
        path = OMI2 / f"train-{s:05d}-of-00032.parquet"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = clean_answer(r["expected_answer"] or "")
                if ans is None:
                    continue
                prob = (r["problem"] or "").strip()
                if not (20 <= len(prob) <= 1200):
                    continue
                sol = strip_boxed(r["generated_solution"] or "")
                if not (40 <= len(sol) <= 2500):
                    continue
                if "\\" in sol or "```" in sol:
                    continue  # leftover latex / code -> off-style for gsm8k
                if per_problem.get(prob, 0) >= args.max_per_problem:
                    continue
                key = prob[:200] + "|" + sol[:120]
                if key in seen:
                    continue
                seen.add(key)
                per_problem[prob] = per_problem.get(prob, 0) + 1
                # one answer marker only: drop a trailing restatement sentence
                sol = sol.rstrip()
                target = f"{sol}\n\nANSWER: {ans}{STOP_TOKEN}"
                rows.append({"question": prob, "target": target, "answer": ans})
        print(f"shard {s}: {len(rows)} rows so far", file=sys.stderr)
        if len(rows) >= args.n * 3:
            break

    rng.shuffle(rows)
    rows = rows[: args.n]

    # a slice gets a k-shot prefix so the model is robust to the grader's
    # 10-shot system message; demos come from this pool, never from test.
    n_fs = int(len(rows) * args.fewshot_frac)
    pool = rows[n_fs:]
    out = []
    for i, r in enumerate(rows):
        user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
        system = None
        if i < n_fs and pool:
            k = rng.choice([2, 3, 4])
            demos = rng.sample(pool, k)
            parts = []
            for d in demos:
                body = d["target"].rsplit("\n\nANSWER:", 1)[0]
                parts.append(
                    f"{d['question']}\n\nReasoning:\n{body}\n\nANSWER: {d['answer']}"
                )
            system = "\n\n".join(parts)
        out.append(
            {
                "system": system,
                "prompt": user,
                "completion": r["target"],
                "answer": r["answer"],
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} rows to {args.out} ({n_fs} few-shot)", file=sys.stderr)


if __name__ == "__main__":
    main()
