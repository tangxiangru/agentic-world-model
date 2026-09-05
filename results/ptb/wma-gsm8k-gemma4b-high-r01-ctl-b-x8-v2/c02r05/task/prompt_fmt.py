"""Prompt rendering that is byte-identical to what the grader sends.

The grader (evaluate.py) hands vLLM `templates/gemma3.jinja` and the
inspect_evals gsm8k task's own MATH_PROMPT_TEMPLATE / 10-shot system message.
Everything here is rendered through those same two artefacts so training text
and grading text cannot drift (pitfall: template_unreachable).
"""
from __future__ import annotations

import hashlib
import os

from jinja2 import Environment
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(HERE, "templates", "gemma3.jinja")
FEWSHOT_PATH = os.path.join(HERE, "data", "fewshot_system.txt")

# inspect_evals.gsm8k.gsm8k.MATH_PROMPT_TEMPLATE, copied verbatim
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
END_OF_TURN = "<end_of_turn>"


def _raise_exception(msg: str):
    raise TemplateError(msg)


def _env() -> Environment:
    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = _raise_exception
    return env


with open(TEMPLATE_PATH, "rb") as _f:
    TEMPLATE_SRC_BYTES = _f.read()
TEMPLATE_SHA = hashlib.sha256(TEMPLATE_SRC_BYTES).hexdigest()
_TEMPLATE = _env().from_string(TEMPLATE_SRC_BYTES.decode("utf-8"))

with open(FEWSHOT_PATH) as _f:
    FEWSHOT_SYSTEM = _f.read()


def render_prompt(question: str, system: str | None = None) -> str:
    """Exactly what vLLM receives as the prompt string, including <bos>."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)}
    )
    return _TEMPLATE.render(
        messages=messages, add_generation_prompt=True, bos_token=BOS
    )


def render_row(question: str, solution: str, system: str | None = None):
    """(prompt, completion) where prompt+completion is the full training text."""
    return render_prompt(question, system), solution.strip() + END_OF_TURN
