#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Sources (both derived from the GSM8K *train* split or from public augmentations of
it; the GSM8K test split is never read here):
  A) openai/gsm8k  split=train        - 7473 human-written CoT solutions
  B) nvidia/OpenMathInstruct-2        - rows with problem_source in {gsm8k, augmented_gsm8k}

Every target is shaped for the harness grader:
  <reasoning>

  ANSWER: <number>
and is terminated with <end_of_turn> by the trainer (token 106), which is what
vLLM stops on for this checkpoint (generation_config.json eos_token_id [1, 106]).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from datasets import load_dataset, load_from_disk

# The prompt the harness wraps every question in (inspect_evals/gsm8k/gsm8k.py).
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUMERIC = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
CALC = re.compile(r"<<[^>]*>>")


def unwrap_boxed(text: str) -> str | None:
    """Replace every \\boxed{...} with its contents. None if braces don't balance."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        if depth:
            return None
        out.append(text[j + len("\\boxed{"): k - 1])
        i = k


def norm_answer(a: str) -> str:
    a = a.strip().replace(",", "")
    if a.endswith(".0"):
        a = a[:-2]
    return a


def make_row(question: str, reasoning: str, answer: str, src: str) -> dict | None:
    reasoning = reasoning.strip()
    if not reasoning:
        return None
    ans = norm_answer(answer)
    if not NUMERIC.match(ans):
        return None
    completion = f"{reasoning}\n\nANSWER: {ans}<end_of_turn>"
    if completion.count("ANSWER:") != 1:
        return None
    return {"question": question.strip(), "completion": completion, "answer": ans, "src": src}


def build_gsm8k_train() -> list[dict]:
    d = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in d:
        body, _, tail = r["answer"].rpartition("####")
        reasoning = CALC.sub("", body).strip()
        row = make_row(r["question"], reasoning, tail, "gsm8k_train")
        if row:
            rows.append(row)
    return rows


def build_omi2(path: str, per_problem: int, seed: int) -> list[dict]:
    d = load_from_disk(path)
    by_problem: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for r in d:
        sol = unwrap_boxed(r["generated_solution"])
        if sol is None or "\\boxed" in sol:
            continue
        if len(sol) > 2600:
            continue
        row = make_row(r["problem"], sol, r["expected_answer"], r["problem_source"])
        if row is None:
            continue
        key = (row["question"], row["completion"])
        if key in seen:
            continue
        seen.add(key)
        by_problem[row["question"]].append(row)
    rng = random.Random(seed)
    out = []
    for _, group in by_problem.items():
        rng.shuffle(group)
        out.extend(group[:per_problem])
    return out


def sample_to_fewshot(q: str, reasoning: str, ans: str) -> str:
    """Byte-identical to inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {ans}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi2-dir", default="/home/ben/task/data/omi2_gsm_1M")
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm = build_gsm8k_train()
    print(f"gsm8k train usable: {len(gsm)}")
    omi = build_omi2(args.omi2_dir, args.per_problem, args.seed)
    print(f"omi2 gsm usable:    {len(omi)}  (<= {args.per_problem} per problem)")

    rows = omi + gsm * args.gsm8k_repeat
    rng.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]

    # few-shot prefix pool: harness-format demonstrations built from GSM8K *train*
    pool = [sample_to_fewshot(r["question"], r["completion"].rsplit("\n\nANSWER:", 1)[0], r["answer"])
            for r in gsm]

    n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            prompt = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.choice([1, 2, 3])
                system = "\n\n".join(rng.sample(pool, k))
                n_fs += 1
            f.write(json.dumps({
                "system": system,
                "prompt": prompt,
                "completion": r["completion"],
                "answer": r["answer"],
                "src": r["src"],
            }) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} with a few-shot prefix)")


if __name__ == "__main__":
    main()
