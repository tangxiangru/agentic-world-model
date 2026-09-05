#!/usr/bin/env python3
"""Build SFT data for GSM8K in exactly the format the grader renders.

Output jsonl rows: {"prompt": <str>, "completion": <str>, "src": <str>, "answer": <str>}

prompt      : the *rendered* prompt string, byte-identical to what templates/gemma3.jinja
              produces for the grader's message list (up to and including
              "<start_of_turn>model\n").
completion  : reasoning, then "ANSWER: <n>", then "<end_of_turn>".
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import Counter

import pyarrow.parquet as pq
from datasets import load_dataset

TASK_DIR = "/home/ben/task"
TEMPLATE = f"{TASK_DIR}/templates/gemma3.jinja"

# copied byte-for-byte from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"


def render_prompt(question: str, fewshot_block: str | None) -> str:
    """Reproduce templates/gemma3.jinja for [system?, user] + add_generation_prompt."""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    prefix = (fewshot_block + "\n\n") if fewshot_block else ""
    return f"{BOS}{SOT}user\n{prefix}{user}{EOT}\n{SOT}model\n"


CALC = re.compile(r"<<[^>]*>>")


def gold_fewshot(sample: dict) -> str:
    """Byte-identical to inspect_evals.gsm8k.sample_to_fewshot(record_to_sample(r)).

    Note it keeps gsm8k's <<48/2=24>> calculator annotations: the grader's own
    10-shot prefix has them, so a training prefix without them would be a
    different context from the one the model is graded in. Targets stay clean.
    """
    reasoning, target = sample["answer"].split("####")
    return f"{sample['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}"


BOXED = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    while True:
        m = BOXED.search(text)
        if not m:
            return text
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth:  # unbalanced, give up
            return text
        text = text[: m.start()] + text[m.end() : i - 1] + text[i:]


TAIL_PATTERNS = [
    re.compile(r"\n*\s*(so\s+)?the (final )?answer is[^\n]*$", re.I),
    re.compile(r"\n*\s*the answer is:[^\n]*$", re.I),
]

INT_RE = re.compile(r"^-?\d+$")


def clean_solution(sol: str, answer: str) -> str | None:
    sol = strip_boxed(sol).strip()
    sol = CALC.sub("", sol)
    if "####" in sol or "ANSWER:" in sol.upper():
        return None
    for pat in TAIL_PATTERNS:
        sol = pat.sub("", sol).strip()
    sol = re.sub(r"[ \t]+\n", "\n", sol).strip()
    if len(sol) < 20:
        return None
    return f"{sol}\n\nANSWER: {answer}"


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-aug-gsm8k", type=int, default=70000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-aug-math", type=int, default=6000)
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    fewshot_pool = [gold_fewshot(s) for s in gsm_train]
    train_questions = {norm_q(s["question"]) for s in gsm_train}

    rows: list[dict] = []

    # ---- 1. original gsm8k train, gold solutions -----------------------------
    for s in gsm_train:
        reasoning, target = s["answer"].split("####")
        reasoning = CALC.sub("", reasoning).strip()
        target = target.strip().replace(",", "")
        rows.append(
            {
                "question": s["question"],
                "completion": f"{reasoning}\n\nANSWER: {target}",
                "src": "gsm8k_gold",
                "answer": target,
            }
        )

    # ---- 2. OpenMathInstruct-2, gsm8k + augmented_gsm8k ----------------------
    files = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train_1M-*.parquet"
        )
    )
    per_problem: Counter = Counter()
    aug_rows: list[dict] = []
    math_rows: list[dict] = []
    for f in files:
        tbl = pq.read_table(f).to_pylist()
        for r in tbl:
            src = r["problem_source"]
            ans = (r["expected_answer"] or "").strip().replace(",", "")
            if src in ("gsm8k", "augmented_gsm8k"):
                if not INT_RE.match(ans):
                    continue
                key = norm_q(r["problem"])
                if per_problem[key] >= args.max_per_problem:
                    continue
                sol = clean_solution(r["generated_solution"], ans)
                if sol is None:
                    continue
                per_problem[key] += 1
                aug_rows.append(
                    {"question": r["problem"], "completion": sol, "src": src, "answer": ans}
                )
            elif src in ("math", "augmented_math") and len(math_rows) < args.n_aug_math * 4:
                if not INT_RE.match(ans):
                    continue
                sol = clean_solution(r["generated_solution"], ans)
                if sol is None:
                    continue
                math_rows.append(
                    {"question": r["problem"], "completion": sol, "src": src, "answer": ans}
                )

    rng.shuffle(aug_rows)
    rows.extend(aug_rows[: args.max_aug_gsm8k])
    rng.shuffle(math_rows)
    rows.extend(math_rows[: args.n_aug_math])

    # ---- 3. render prompts, some with a few-shot prefix ----------------------
    out = []
    seen = set()
    for r in rows:
        h = hashlib.md5((norm_q(r["question"]) + "|" + r["completion"]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        if rng.random() < args.fewshot_frac:
            k = rng.choice([1, 2, 3, 4, 5, 8])
            shots = rng.sample(fewshot_pool, k)
            block = "\n\n".join(shots)
        else:
            block = None
        out.append(
            {
                "prompt": render_prompt(r["question"], block),
                "completion": r["completion"] + EOT,
                "src": r["src"],
                "answer": r["answer"],
            }
        )

    rng.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(out)} rows to {args.out}")
    print(Counter(r["src"] for r in out))
    print("train-question pool:", len(train_questions))


if __name__ == "__main__":
    main()
