#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Targets are shaped for the grader in inspect_evals/gsm8k:
  * user turn   = MATH_PROMPT_TEMPLATE (verbatim copy of the grader's) around the question
  * model turn  = chain of thought, then a final line "ANSWER: <n>"
  * terminator  = <end_of_turn>, the token templates/gemma3.jinja closes a turn with
The grader scores with match(numeric=True, location="end"): the LAST number anywhere
in the completion must be the gold answer, so the answer line is always last and the
chain of thought never trails a number after it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

# --- verbatim from inspect_evals/gsm8k/gsm8k.py -----------------------------
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANSWER_MARKER = "ANSWER: "
# the terminator templates/gemma3.jinja closes an assistant turn with, and one of the
# two ids vLLM stops on (generation_config eos_token_id [1, 106]). It is written into
# the stored target so the corpus on disk is what the trainer feeds the model.
STOP_TOKEN = "<end_of_turn>"

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"

_CALC = re.compile(r"<<[^>]*>>")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def clean_number(s: str, integers_only: bool = False) -> str | None:
    """Normalise a gold answer to a bare number string, or None if it is not one.

    integers_only mirrors the benchmark: every gsm8k gold answer is an integer, and
    the non-integer augmented rows are where the rambling round-to-2-dp solutions live.
    """
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not _NUM.fullmatch(s):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s.startswith("."):
        s = "0" + s
    if integers_only and not re.fullmatch(r"-?\d+", s):
        return None
    return s


def strip_boxed(text: str) -> str:
    """Unwrap \\boxed{x} -> x; the grader is told the model need not use \\boxed."""
    prev = None
    while prev != text:
        prev = text
        text = _BOXED.sub(r"\1", text)
    return text.replace("\\boxed", "")


def make_target(solution: str, answer: str) -> str | None:
    body = strip_boxed(_CALC.sub("", solution)).strip()
    if not body:
        return None
    # drop any pre-existing answer marker so ANSWER: appears exactly once
    body = re.sub(r"(?im)^\s*(####|ANSWER:|The answer is:?)\s*.*$", "", body).strip()
    if not body:
        return None
    return f"{body}\n\n{ANSWER_MARKER}{answer}{STOP_TOKEN}"


MAX_TARGET_CHARS = 1800   # ~450 tokens; longer omi2 rows are almost all self-corrections that ramble


def load_omi2(shards: int, sources: set[str], max_per_problem: int, cap: int, rng: random.Random):
    rows, per_problem = [], {}
    for i in range(shards):
        f = Path(OMI2) / f"train-{i:05d}-of-00032.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df = df[df["problem_source"].isin(sources)]
        for problem, solution, expected in zip(df["problem"], df["generated_solution"], df["expected_answer"]):
            ans = clean_number(expected, integers_only=True)
            if ans is None or len(solution) > MAX_TARGET_CHARS:
                continue
            k = per_problem.get(problem, 0)
            if k >= max_per_problem:
                continue
            tgt = make_target(solution, ans)
            if tgt is None:
                continue
            per_problem[problem] = k + 1
            rows.append({"question": problem.strip(), "target": tgt, "answer": ans})
        if len(rows) >= cap * 2:
            break   # enough to shuffle from; reading further shards only costs time
    rng.shuffle(rows)
    print(f"[omi2] {sorted(sources)}: {len(rows)} rows over {len(per_problem)} unique problems -> keeping {min(cap, len(rows))}")
    return rows[:cap]


def load_gsm8k_train(rng: random.Random):
    df = pd.read_parquet(GSM8K_TRAIN)
    rows = []
    for q, a in zip(df["question"], df["answer"]):
        body, _, gold = a.partition("####")
        ans = clean_number(gold)
        if ans is None:
            continue
        tgt = make_target(body, ans)
        if tgt is None:
            continue
        rows.append({"question": q.strip(), "target": tgt, "answer": ans})
    rng.shuffle(rows)
    return rows


def fewshot_block(row: dict) -> str:
    """One shot rendered the way the grader renders its own 10 shots."""
    body, _, ans = row["target"].removesuffix(STOP_TOKEN).rpartition(f"\n\n{ANSWER_MARKER}")
    return f"{row['question']}\n\nReasoning:\n{body}\n\n{ANSWER_MARKER}{ans}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k-omi2", type=int, default=44000)
    ap.add_argument("--n-math-omi2", type=int, default=6000)
    ap.add_argument("--gsm8k-train-repeat", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="already-formatted jsonl (e.g. rejection-sampled rows) to mix in verbatim")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm_omi2 = load_omi2(args.shards, {"gsm8k", "augmented_gsm8k"}, args.max_per_problem, args.n_gsm8k_omi2, rng)
    math_omi2 = load_omi2(args.shards, {"math", "augmented_math"}, 1, args.n_math_omi2, rng)
    gsm_orig = load_gsm8k_train(rng)
    extra = []
    for path in args.extra:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                extra.append({"question": r["question"], "target": r["completion"], "answer": r["answer"]})
        print(f"[extra] {path}: {len(extra)} rows cumulative")

    pool = gsm_omi2 + math_omi2 + gsm_orig * args.gsm8k_train_repeat + extra
    rng.shuffle(pool)
    print(f"omi2-gsm8k {len(gsm_omi2)}  omi2-math {len(math_omi2)}  gsm8k-train x{args.gsm8k_train_repeat} "
          f"{len(gsm_orig) * args.gsm8k_train_repeat}  extra {len(extra)}  total {len(pool)}")

    # a slice of rows gets a few-shot prefix so the model sees the long-prompt
    # condition it is graded under and does not copy the shots' terse style
    shot_pool = gsm_orig
    n_fewshot = int(len(pool) * args.fewshot_frac)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, row in enumerate(pool):
            prefix = ""
            if i < n_fewshot:
                k = rng.choice([2, 4, 8, 10])
                shots = rng.sample(shot_pool, k)
                if all(s["question"] != row["question"] for s in shots):
                    prefix = "\n\n".join(fewshot_block(s) for s in shots) + "\n\n"
            f.write(json.dumps({
                "prompt": prefix + MATH_PROMPT_TEMPLATE.format(prompt=row["question"]),
                "completion": row["target"],
                "question": row["question"],
                "answer": row["answer"],
            }) + "\n")
    print(f"wrote {len(pool)} rows to {out} ({n_fewshot} with a few-shot prefix)")


if __name__ == "__main__":
    main()
