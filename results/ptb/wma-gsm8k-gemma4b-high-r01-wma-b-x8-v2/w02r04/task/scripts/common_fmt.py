"""Shared formatting helpers: render exactly what the grader renders.

The grader (evaluate.py) passes templates/gemma3.jinja to vLLM and builds the
prompt with inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE plus a 10-shot system
message. Everything here is derived from those two files so training and
grading cannot drift.
"""
from __future__ import annotations

import hashlib
import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# byte-for-byte copy of inspect_evals/gsm8k/gsm8k.py::MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"
STOP_TOKEN = EOT


def template_sha() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question.strip())


def render_prompt(question: str, system: str | None = None) -> str:
    """The exact string the gemma3.jinja template produces up to the model turn."""
    first_user_prefix = (system + "\n\n") if system else ""
    content = user_prompt(question).strip()
    return (
        f"{BOS}{SOT}user\n{first_user_prefix}{content}{EOT}\n{SOT}model\n"
    )


def render_target(solution: str) -> str:
    """Assistant turn content plus the terminator the grader stops on."""
    return f"{solution.strip()}{EOT}\n"


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def fmt_number(x: str) -> str | None:
    x = str(x).strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.fullmatch(r"-?\d+(\.\d+)?", x)
    if not m:
        return None
    if "." in x:
        f = float(x)
        if f == int(f):
            return str(int(f))
        return ("%.10g" % f)
    return str(int(x))
