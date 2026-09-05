#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Every row is {"prompt": <full user-turn content>, "completion": <model turn,
ending in the grader's stop token>}. The prompt is rendered with the same
MATH_PROMPT_TEMPLATE the inspect_evals/gsm8k solver uses, and a share of rows
carry a k-shot prefix in the harness's own few-shot format so the model sees
the 10-shot eval context shape at training time.

Sources (both derived from the GSM8K *train* split only):
  * nvidia/OpenMathInstruct-2, train_1M, problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k train split, original human solutions
"""
import argparse
import glob
import json
import random
import re

import pandas as pd

STOP = "<end_of_turn>"
OMI2_GLOB = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
    "snapshots/*/data/train_1M-*.parquet"
)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
CALC_RE = re.compile(r"<<[^>]*>>")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def strip_boxed(text: str) -> str:
    """\\boxed{14} -> 14 ; also drop a stray \\[ \\] display wrapper around it."""
    text = BOXED_RE.sub(r"\1", text)
    text = text.replace("\\[", "").replace("\\]", "")
    return text.strip()


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def make_row(question: str, solution: str, answer: str) -> dict | None:
    solution = solution.strip()
    if "ANSWER:" in solution or "answer:" in solution.lower():
        return None
    completion = f"{solution}\n\nANSWER: {answer}{STOP}"
    return {"question": question.strip(), "solution": solution, "answer": answer,
            "completion": completion}


def gsm8k_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        q = r["question"]
        parts = r["answer"].split("####")
        ans = norm_answer(parts[-1])
        if ans is None:
            continue
        reasoning = CALC_RE.sub("", "####".join(parts[:-1])).strip()
        reasoning = re.sub(r"[ \t]+", " ", reasoning)
        row = make_row(q, reasoning, ans)
        if row:
            row["src"] = "gsm8k_train"
            out.append(row)
    return out


def omi2_gsm8k(max_per_problem: int, rng: random.Random):
    files = sorted(glob.glob(OMI2_GLOB))
    assert files, "OpenMathInstruct-2 parquet files not found"
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "generated_solution",
                                         "expected_answer", "problem_source"])
        frames.append(df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])])
    df = pd.concat(frames, ignore_index=True)
    print(f"omi2 gsm8k-family rows: {len(df)}")

    seen: dict[str, int] = {}
    out = []
    idx = list(range(len(df)))
    rng.shuffle(idx)
    for i in idx:
        r = df.iloc[i]
        ans = norm_answer(str(r.expected_answer))
        if ans is None:
            continue
        q = r.problem.strip()
        if seen.get(q, 0) >= max_per_problem:
            continue
        sol = strip_boxed(r.generated_solution)
        if not sol or len(sol) < 30:
            continue
        row = make_row(q, sol, ans)
        if row is None:
            continue
        seen[q] = seen.get(q, 0) + 1
        row["src"] = r.problem_source
        out.append(row)
    print(f"omi2 kept: {len(out)} over {len(seen)} distinct problems")
    return out


def fewshot_block(rows, k: int, rng: random.Random) -> str:
    """k demonstrations in the exact format inspect_evals' sample_to_fewshot emits."""
    picks = rng.sample(rows, k)
    return "\n\n".join(
        f"{p['question']}\n\nReasoning:\n{p['solution']}\n\nANSWER: {p['answer']}"
        for p in picks
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-omi2", type=int, default=52000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--decon-out", default="/home/ben/task/data/sft_v1.decon.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm = gsm8k_train()
    print(f"gsm8k train rows: {len(gsm)}")
    omi = omi2_gsm8k(args.max_per_problem, rng)
    rng.shuffle(omi)
    omi = omi[: args.n_omi2]

    pool = gsm + omi
    rng.shuffle(pool)

    # k-shot demonstrations are drawn from the original gsm8k train solutions,
    # which is what the harness itself uses for its 10-shot prefix.
    n_fs = int(len(pool) * args.fewshot_frac)
    with open(args.out, "w") as f, open(args.decon_out, "w") as g:
        for i, row in enumerate(pool):
            user = MATH_PROMPT_TEMPLATE.format(prompt=row["question"])
            if i < n_fs:
                k = rng.choice([2, 3, 4, 6, 8])
                user = fewshot_block(gsm, k, rng) + "\n\n" + user
            f.write(json.dumps({"prompt": user, "completion": row["completion"],
                                "src": row["src"]}) + "\n")
            g.write(json.dumps({"question": row["question"],
                                "answer": row["completion"][: -len(STOP)]}) + "\n")

    print(f"wrote {len(pool)} rows -> {args.out}")
    print(f"  fewshot-prefixed: {n_fs}")
    from collections import Counter
    print(" ", Counter(r["src"] for r in pool))


if __name__ == "__main__":
    main()
