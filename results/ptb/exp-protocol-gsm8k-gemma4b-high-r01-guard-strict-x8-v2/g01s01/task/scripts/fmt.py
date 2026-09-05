"""Shared formatting helpers: renders training rows byte-identically to the grader.

The grader is inspect_evals/gsm8k with templates/gemma3.jinja (see evaluate.py
template_kwargs). Everything here is derived from those two files so that a
training row and a graded prompt are the same string.
"""
from __future__ import annotations

import hashlib
import os
import re

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_sha() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def user_prompt(question: str) -> str:
    """The user-turn content inspect builds: MATH_PROMPT_TEMPLATE around the question."""
    return MATH_PROMPT_TEMPLATE.replace("{prompt}", question)


def render_prompt(question: str, system: str | None = None) -> str:
    """Reproduce templates/gemma3.jinja for [system?, user] + add_generation_prompt.

    gemma3.jinja: bos, then the system content becomes `first_user_prefix`
    (content + '\\n\\n') glued in front of the first user turn's trimmed content,
    then '<end_of_turn>\\n', then '<start_of_turn>model\\n'.
    """
    first_user_prefix = (system + "\n\n") if system else ""
    body = user_prompt(question).strip()
    return (
        f"{BOS}{SOT}user\n{first_user_prefix}{body}{EOT}\n{SOT}model\n"
    )


def render_completion(solution: str) -> str:
    """Model-turn content + the terminator the grader stops on."""
    return f"{solution.strip()}{EOT}"


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_CALC = re.compile(r"<<[^>]*>>")
# GSM8K's own answer marker. MetaMathQA bodies carry it; leaving it in would
# teach a second marker the grader might read instead of ours
# (pitfalls.yaml: double_answer_format).
_HASH = re.compile(r"^[ \t]*#{2,}[ \t]*.*$", re.M)


def clean_solution(text: str) -> str:
    text = _BOXED.sub(r"\1", text)
    text = _CALC.sub("", text)
    text = _HASH.sub("", text)
    text = text.replace("\\dfrac", "\\frac")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_target(solution: str, answer: str) -> str:
    """Body + exactly one 'ANSWER: <n>' final line. Returns None if unusable."""
    body = clean_solution(solution)
    if ANSWER_MARKER.lower() in body.lower():
        return None
    return f"{body}\n\n{ANSWER_MARKER}{answer}"
