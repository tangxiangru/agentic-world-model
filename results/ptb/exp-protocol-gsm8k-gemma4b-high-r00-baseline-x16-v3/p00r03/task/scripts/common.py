"""Shared pieces: the exact eval rendering, the exact grader, dev split."""
from __future__ import annotations

import os
import re
import json

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
GEMMA_TEMPLATE = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# Copied byte-for-byte from inspect_evals/gsm8k/gsm8k.py (MATH_PROMPT_TEMPLATE).
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_system_message(n: int = 10, seed: int = 42):
    """Reproduce inspect_evals' fewshot system message exactly."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    msg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])
    questions = [s.input for s in fewshots]
    return msg, questions


# ---- grading: replicates inspect_ai match(numeric=True, location="end") ----
from inspect_ai.scorer._common import match_str  # noqa: E402


def grade(completion: str, target: str) -> bool:
    _, ok = match_str(completion, target, location="end", ignore_case=True, numeric=True)
    return bool(ok)


def gsm8k_gold(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


CALC_RE = re.compile(r"<<[^>]*>>")


def strip_calc(text: str) -> str:
    return CALC_RE.sub("", text)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
