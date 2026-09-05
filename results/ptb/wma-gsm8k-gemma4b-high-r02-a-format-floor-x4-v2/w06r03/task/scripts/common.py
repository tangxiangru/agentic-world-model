"""Shared helpers: the exact grading prompt, the exact chat template, the exact scorer.

Everything here is copied from what the grader actually runs so that training and
grading render the same strings (pitfalls.yaml: template_unreachable, eos_mismatch,
double_answer_format).
"""
from __future__ import annotations

import hashlib
import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# inspect_evals/gsm8k MATH_PROMPT_TEMPLATE, byte-for-byte
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"


def load_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def sample_to_fewshot(question: str, reasoning: str, target: str) -> str:
    """inspect_evals/gsm8k sample_to_fewshot, byte-for-byte."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


# ---------------------------------------------------------------- scorer copy
# inspect_ai.scorer._common.match_str with location="end", numeric=True

def _strip_numeric_punctuation(s: str) -> str:
    stripped = re.sub(r"[$,£,€,*,_]", "", s)
    stripped = re.sub(r"\.(?=\s|$|\D)", "", stripped)
    return stripped


def _normalize_number(number: str, precision: int = 5) -> str:
    if number.replace(".", "").isnumeric():
        try:
            m = re.match(r"^([+-]?\d+(?:\.\d+)?)", number)
            num = float(m.group(1)) if m else float(number)
        except (ValueError, AttributeError):
            return number
        return format(num, f".{precision}g")
    return number


def _first_number_normalized(words):
    number = next((w for w in words if w.replace(".", "").isnumeric()), words[0])
    return _normalize_number(number)


def grade(completion: str, target: str) -> bool:
    """True iff inspect's match(numeric=True) would score this CORRECT."""
    v = completion.strip().casefold()
    t = target.strip().casefold()
    if not t.isnumeric():
        # non-numeric targets fall back to punctuation-stripped endswith
        return v.endswith(t)
    v = _strip_numeric_punctuation(v)
    t = _normalize_number(_strip_numeric_punctuation(t))
    words = re.split(r"\s+", v)
    words.reverse()
    if not words:
        return False
    v = _first_number_normalized(words)
    return v.endswith(t)
