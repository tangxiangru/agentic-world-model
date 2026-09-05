#!/usr/bin/env python3
"""Build the SFT mixture for gemma-3-4b-pt -> GSM8K.

Target format is dictated by the grader (inspect_evals/gsm8k):
  user turn   = MATH_PROMPT_TEMPLATE.format(prompt=question)   (ends with "Reasoning:")
  model turn  = <chain of thought>\n\nANSWER: <number>
  terminator  = <end_of_turn>   (token 106, in generation_config.eos_token_id)

Sources (all GSM8K/MATH *train*-derived; the benchmark test split is never read):
  - openai/gsm8k          split=train      human-written CoT
  - nvidia/OpenMathInstruct-2               augmented_gsm8k / gsm8k / augmented_math
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
BOXED_TAIL = re.compile(
    r"(?:so |thus |therefore )?the (?:final )?answer is[^.]*\\boxed\{[^}]*\}\$?\.?\s*$",
    re.IGNORECASE,
)


def clean_gsm8k_answer(ans: str) -> tuple[str, str] | None:
    if "####" not in ans:
        return None
    body, final = ans.rsplit("####", 1)
    body = CALC.sub("", body).strip()
    final = final.strip().replace(",", "")
    if not body or not final:
        return None
    return body, final


def clean_omi_solution(sol: str, expected: str) -> str | None:
    """Drop the trailing 'The answer is \\boxed{...}' sentence; keep the derivation."""
    s = sol.strip()
    # remove a trailing standalone boxed statement so 'ANSWER:' is the only verdict
    s2 = BOXED_TAIL.sub("", s).strip()
    if len(s2) < 20:
        # the whole solution was the boxed line - keep the original
        s2 = s
    # a bare trailing "\[ x = \boxed{2} \]" style line is fine to keep; it is derivation
    s2 = s2.rstrip()
    if not s2:
        return None
    return s2


def render(question: str, target_body: str, answer: str) -> dict:
    return {
        "prompt": MATH_PROMPT_TEMPLATE.format(prompt=question.strip()),
        "target": f"{target_body.strip()}\n\nANSWER: {answer.strip()}",
        "answer": answer.strip(),
    }


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9]", "", q.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_mix.jsonl")
    ap.add_argument("--n-aug-gsm8k", type=int, default=55000)
    ap.add_argument("--n-omi-gsm8k", type=int, default=12000)
    ap.add_argument("--n-aug-math", type=int, default=8000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []

    # ---- 1. openai/gsm8k train -------------------------------------------
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    n_gsm = 0
    for r in gsm:
        c = clean_gsm8k_answer(r["answer"])
        if c is None:
            continue
        body, final = c
        row = render(r["question"], body, final)
        row["src"] = "gsm8k_train"
        for _ in range(args.gsm8k_repeat):
            rows.append(dict(row))
        n_gsm += 1
    print(f"gsm8k_train: {n_gsm} unique x{args.gsm8k_repeat}")

    # ---- 2. OpenMathInstruct-2 -------------------------------------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    print("omi loaded", omi)
    want = {
        "augmented_gsm8k": args.n_aug_gsm8k,
        "gsm8k": args.n_omi_gsm8k,
        "augmented_math": args.n_aug_math,
    }
    taken = defaultdict(int)
    per_problem: dict[str, int] = defaultdict(int)
    omi = omi.select_columns(
        ["problem", "generated_solution", "expected_answer", "problem_source"]
    )
    for src, quota in want.items():
        sub = omi.filter(
            lambda b: [s == src for s in b["problem_source"]],
            batched=True,
            num_proc=8,
        )
        sub = sub.shuffle(seed=args.seed)
        # over-select: some rows are dropped by the filters below
        sub = sub.select(range(min(len(sub), int(quota * 3) + 1000)))
        for r in sub:
            if taken[src] >= quota:
                break
            key = norm_q(r["problem"])
            if per_problem[key] >= args.max_per_problem:
                continue
            ans = str(r["expected_answer"]).strip()
            if not ans or len(ans) > 30:
                continue
            body = clean_omi_solution(r["generated_solution"], ans)
            if body is None:
                continue
            row = render(r["problem"], body, ans)
            row["src"] = src
            rows.append(row)
            taken[src] += 1
            per_problem[key] += 1
        print(f"  {src}: {taken[src]} / {quota} (pool {len(sub)})")
    print("omi taken:", dict(taken))

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
