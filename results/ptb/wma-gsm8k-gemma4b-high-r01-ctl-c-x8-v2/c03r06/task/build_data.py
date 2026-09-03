#!/usr/bin/env python3
"""Build SFT data for GSM8K, rendered byte-for-byte in the grader's format.

Sources (all train-split / train-derived, never the benchmark test split):
  - openai/gsm8k          split=train   (human-written reference solutions)
  - nvidia/OpenMathInstruct-2 train_1M, problem_source in {gsm8k, augmented_gsm8k}

Output: jsonl with {prompt, completion, ...} where
  prompt     = the exact string templates/gemma3.jinja renders for the grader's
               conversation, up to and including "<start_of_turn>model\n"
  completion = reasoning + "\n\nANSWER: <n>" + "<end_of_turn>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# ---- the grader's prompt, copied verbatim from inspect_evals/gsm8k/gsm8k.py ----
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI_REV = "469216e3f46f4dacf476b382e192485ea51a143e"
GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"


# ---- rendering (mirrors templates/gemma3.jinja) -------------------------------
def render_prompt(system: str | None, user: str) -> str:
    """Exactly what templates/gemma3.jinja produces for [system?, user] with
    add_generation_prompt=True. Verified byte-for-byte in check_template.py."""
    first_user_prefix = (system.strip() + "\n\n") if system else ""
    return (
        "<bos><start_of_turn>user\n"
        + first_user_prefix
        + user.strip()
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def fewshot_block(q: str, reasoning: str, answer: str) -> str:
    """inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


# ---- cleaning -----------------------------------------------------------------
CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Replace every \boxed{...} with its contents (brace-balanced)."""
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
        if depth:
            return text[: m.start()] + text[m.end():]
        text = text[: m.start()] + text[m.end(): i - 1] + text[i:]


def norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    return s


def looks_numeric(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


def gsm8k_train_rows():
    fn = hf_hub_download(
        "openai/gsm8k", "main/train-00000-of-00001.parquet",
        repo_type="dataset", revision=GSM8K_REV,
    )
    for r in pq.read_table(fn).to_pylist():
        body, _, ans = r["answer"].rpartition("####")
        body = CALC.sub("", body).strip()
        ans = norm_num(ans)
        if not looks_numeric(ans):
            continue
        yield {"q": r["question"].strip(), "sol": body, "ans": ans, "src": "gsm8k_train"}


def omi_rows(max_per_problem: int):
    seen = defaultdict(int)
    for i in range(3):
        fn = hf_hub_download(
            "nvidia/OpenMathInstruct-2", f"data/train_1M-0000{i}-of-00003.parquet",
            repo_type="dataset", revision=OMI_REV,
        )
        for r in pq.read_table(fn).to_pylist():
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = norm_num(r["expected_answer"] or "")
            if not looks_numeric(ans):
                continue
            key = hashlib.md5(r["problem"].strip().encode()).hexdigest()
            if seen[key] >= max_per_problem:
                continue
            sol = strip_boxed(r["generated_solution"]).strip()
            if len(sol) < 30 or len(sol) > 3500:
                continue
            # the appended ANSWER line must be the last number: drop solutions
            # that trail off after the answer with unrelated arithmetic
            seen[key] += 1
            yield {"q": r["problem"].strip(), "sol": sol, "ans": ans,
                   "src": r["problem_source"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--n-omi", type=int, default=55000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pool = []
    gsm = list(gsm8k_train_rows())
    print(f"gsm8k train usable: {len(gsm)}")
    for _ in range(args.gsm8k_repeat):
        pool.extend(gsm)

    omi = list(omi_rows(args.max_per_problem))
    print(f"OpenMathInstruct-2 gsm8k-family usable: {len(omi)}")
    rng.shuffle(omi)
    pool.extend(omi[: args.n_omi])

    # few-shot exemplars for the robustness slice: gsm8k TRAIN rows only
    shot_pool = gsm[: 2000]

    rng.shuffle(pool)
    n_written = 0
    with open(args.out, "w") as f:
        for ex in pool:
            user = MATH_PROMPT_TEMPLATE.format(prompt=ex["q"])
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 3, 4, 5, 8, 10])
                shots = rng.sample(shot_pool, k)
                system = "\n\n".join(
                    fewshot_block(s["q"], s["sol"], s["ans"]) for s in shots
                )
            prompt = render_prompt(system, user)
            completion = f"{ex['sol']}\n\nANSWER: {ex['ans']}<end_of_turn>"
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "source": ex["src"],
                "answer": ex["ans"],
                "question": ex["q"],
            }) + "\n")
            n_written += 1
    print(f"wrote {n_written} rows to {args.out}")


if __name__ == "__main__":
    main()
