#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (gsm8k-derived subsets) + GSM8K train.

Target format is the one the grader reads:
    <reasoning>
    ANSWER: <number>
rendered through templates/gemma3.jinja, so every target ends with <end_of_turn>.

Nothing here touches the GSM8K test split.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED = re.compile(r"\\boxed\{")


def unbox(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    while True:
        m = BOXED.search(text)
        if m is None:
            return text
        i = m.end()  # first char inside the brace
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth:  # unbalanced; give up rather than corrupt the row
            return text.replace("\\boxed{", "")
        text = text[: m.start()] + text[m.end() : i - 1] + text[i:]


TAIL_ANSWER_SENT = re.compile(
    r"(?:\n|^)[^\n]*?(?:final answer is|answer is|the answer:)[^\n]*$", re.IGNORECASE
)


def clean_solution(sol: str) -> str:
    sol = unbox(sol.strip())
    # kill LaTeX display-math wrappers that only add noise for word problems
    sol = sol.replace("\\[", "").replace("\\]", "")
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def is_plain_number(s: str) -> bool:
    s = s.strip().replace(",", "")
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


def build_target(sol: str, ans: str) -> str | None:
    body = clean_solution(sol)
    if not body:
        return None
    # drop a trailing "The final answer is N" style sentence so exactly one
    # answer marker survives (pitfalls.yaml: double_answer_format)
    body = TAIL_ANSWER_SENT.sub("", body).strip()
    if not body:
        return None
    ans = ans.strip().replace(",", "")
    if ans.endswith(".0"):
        ans = ans[:-2]
    return f"{body}\nANSWER: {ans}"


def gsm8k_train_rows():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        q = r["question"].strip()
        body, ans = r["answer"].rsplit("####", 1)
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        yield q, f"{body}\nANSWER: {ans.strip().replace(',', '')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi-gsm8k", type=int, default=60000)
    ap.add_argument("--n-omi-math", type=int, default=0)
    ap.add_argument("--gsm8k-train-repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sol-chars", type=int, default=3000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(glob.glob(OMI2))
    assert files, OMI2

    gsm_rows: list[tuple[str, str]] = []
    math_rows: list[tuple[str, str]] = []
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        d = t.to_pydict()
        for q, sol, ans, src in zip(
            d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
        ):
            if len(sol) > args.max_sol_chars:
                continue
            if not is_plain_number(ans):
                continue
            tgt = build_target(sol, ans)
            if tgt is None or "ANSWER:" not in tgt:
                continue
            if tgt.count("ANSWER:") != 1:
                continue
            if src in ("gsm8k", "augmented_gsm8k"):
                gsm_rows.append((q.strip(), tgt))
            elif args.n_omi_math:
                math_rows.append((q.strip(), tgt))
        del t, d

    print(f"omi2 gsm8k-family rows: {len(gsm_rows)}   math-family rows: {len(math_rows)}")

    # dedup on (question, target)
    def dedup(rows):
        seen, out = set(), []
        for q, a in rows:
            k = (q, a)
            if k in seen:
                continue
            seen.add(k)
            out.append((q, a))
        return out

    gsm_rows = dedup(gsm_rows)
    rng.shuffle(gsm_rows)
    gsm_rows = gsm_rows[: args.n_omi_gsm8k]
    if args.n_omi_math:
        math_rows = dedup(math_rows)
        rng.shuffle(math_rows)
        math_rows = math_rows[: args.n_omi_math]
    else:
        math_rows = []

    native = list(gsm8k_train_rows())
    print(f"gsm8k train rows: {len(native)}")

    rows = gsm_rows + math_rows + native * args.gsm8k_train_repeat
    rng.shuffle(rows)

    with open(args.out, "w") as fh:
        for q, a in rows:
            fh.write(
                json.dumps(
                    {
                        "prompt": PROMPT_TEMPLATE.format(prompt=q),
                        "completion": a,
                        "question": q,
                        "text": q + "\n" + a,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
