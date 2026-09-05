"""Rendering that is byte-identical to what the grader (evaluate.py) produces.

Sources of truth:
  * templates/gemma3.jinja                       -- the chat template evaluate.py hands to vLLM
  * inspect_evals/gsm8k/gsm8k.py                 -- MATH_PROMPT_TEMPLATE, sample_to_fewshot
  * inspect_ai/scorer/_common.py match_str       -- location="end", numeric=True
"""
from __future__ import annotations

import hashlib
import os
import re

from jinja2 import Environment
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
EOT = "<end_of_turn>"


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _env() -> Environment:
    def raise_exception(msg: str):
        raise TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env


_TEMPLATE = None


def _template():
    global _TEMPLATE
    if _TEMPLATE is None:
        with open(TEMPLATE_PATH) as f:
            _TEMPLATE = _env().from_string(f.read())
    return _TEMPLATE


def render_prompt(question: str, system: str | None = None) -> str:
    """The exact string vLLM receives (chat template applied, generation prompt on)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)}
    )
    return _template().render(
        messages=messages, add_generation_prompt=True, bos_token=BOS
    )


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def fewshot_system(shots: list[tuple[str, str, str]]) -> str:
    return "\n\n".join(fewshot_block(*s) for s in shots)


# ---- grading -----------------------------------------------------------------
# Use the grader's own function rather than a re-implementation: the eval scores
# with match(location="end", numeric=True), which is str_match_scorer over
# inspect_ai.scorer._common.match_str with exactly these arguments.
from inspect_ai.scorer._common import match_str  # noqa: E402


def graded_answer(completion: str) -> str | None:
    """The token inspect's end-anchored numeric matcher would read, or None."""
    answer, _ = match_str(
        value=completion, target="0", location="end", ignore_case=True, numeric=True
    )
    return answer if answer.replace(".", "").replace("-", "").isnumeric() else None


def is_correct(completion: str, target: str) -> bool:
    _, matched = match_str(
        value=completion,
        target=str(target),
        location="end",
        ignore_case=True,
        numeric=True,
    )
    return bool(matched)
