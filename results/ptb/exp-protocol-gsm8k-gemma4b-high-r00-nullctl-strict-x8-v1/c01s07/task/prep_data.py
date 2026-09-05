#!/usr/bin/env python3
"""Build the SFT dataset for GSM8K-style math from OpenMathInstruct-2."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED_RE = re.compile(r"\\boxed\s*\{")


def unbox(text: str) -> str:
    """Replace every \\boxed{...} with its contents (handles nested braces)."""
    while True:
        m = BOXED_RE.search(text)
        if m is None:
            return text
        start = m.end()  # index just after '{'
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth:
            return text[: m.start()] + text[start:]
        text = text[: m.start()] + text[start : i - 1] + text[i:]


def clean_solution(sol: str, answer: str) -> str | None:
    sol = unbox(sol.strip())
    sol = sol.replace("\\[", "").replace("\\]", "")
    sol = re.sub(r"\n{3,}", "\n\n", sol).strip()
    # Drop a redundant trailing "Answer: X" line -- we append our own.
    lines = sol.split("\n")
    while lines and re.fullmatch(
        r"\s*(?:the\s+)?answer\s*(?:is)?\s*[:=]?\s*\$?[^\n]{0,40}", lines[-1], re.I
    ):
        lines.pop()
    sol = "\n".join(lines).strip()
    if not sol:
        return None
    return f"{sol}\n\nANSWER: {answer}"


def load_openmath(sources: dict[str, int], seed: int, max_sol_chars: int):
    files = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train_1M-*.parquet"
        )
    )
    assert files, "OpenMathInstruct-2 shards not found"
    pools: dict[str, list] = {k: [] for k in sources}
    for f in files:
        df = pq.read_table(f).to_pandas()
        for src in sources:
            sub = df[df.problem_source == src]
            pools[src].extend(
                zip(sub["problem"], sub["generated_solution"], sub["expected_answer"])
            )
    rng = random.Random(seed)
    out = []
    seen_problems: set[str] = set()
    for src, n_want in sources.items():
        pool = pools[src]
        rng.shuffle(pool)
        taken = 0
        for problem, sol, ans in pool:
            if taken >= n_want:
                break
            if len(sol) > max_sol_chars or len(problem) > 1500:
                continue
            key = problem.strip().lower()
            if key in seen_problems:
                continue
            body = clean_solution(sol, ans)
            if body is None:
                continue
            seen_problems.add(key)
            out.append({"problem": problem.strip(), "solution": body, "answer": ans, "source": src})
            taken += 1
        print(f"{src}: pool={len(pool)} taken={taken}")
    return out


def gsm8k_fewshot_blocks(n_blocks: int, seed: int):
    """Few-shot blocks in exactly the eval's `sample_to_fewshot` format, built
    from the GSM8K *train* split (never the test split)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    blocks = []
    for i in idx[:n_blocks]:
        rec = ds[i]
        q = rec["question"]
        parts = rec["answer"].split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        blocks.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--n-aug-gsm8k", type=int, default=95000)
    ap.add_argument("--n-gsm8k", type=int, default=15000)
    ap.add_argument("--n-aug-math", type=int, default=12000)
    ap.add_argument("--n-math", type=int, default=3000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--max-sol-chars", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load_openmath(
        {
            "augmented_gsm8k": args.n_aug_gsm8k,
            "gsm8k": args.n_gsm8k,
            "augmented_math": args.n_aug_math,
            "math": args.n_math,
        },
        args.seed,
        args.max_sol_chars,
    )
    rng = random.Random(args.seed + 1)
    rng.shuffle(recs)

    fewshot_pool = gsm8k_fewshot_blocks(600, args.seed)

    with open(args.out, "w") as f:
        for r in recs:
            user = MATH_PROMPT_TEMPLATE.format(prompt=r["problem"])
            system = ""
            if rng.random() < args.fewshot_frac:
                k = rng.randint(2, 10)
                system = "\n\n".join(rng.sample(fewshot_pool, k))
            f.write(
                json.dumps(
                    {
                        "system": system,
                        "user": user,
                        "assistant": r["solution"],
                        "answer": r["answer"],
                        "source": r["source"],
                    }
                )
                + "\n"
            )
    print(f"wrote {len(recs)} -> {args.out}")


if __name__ == "__main__":
    main()
