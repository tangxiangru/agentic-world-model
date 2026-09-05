"""Byte-for-byte reuse of the grader's chat template + prompt strings.

The grader (evaluate.py) passes templates/gemma3.jinja to vLLM and builds its
prompt with inspect_evals/gsm8k's MATH_PROMPT_TEMPLATE and sample_to_fewshot.
Anything that renders a training row must go through this module so training
and grading render the same string (pitfall: template_unreachable).
"""
from __future__ import annotations

import hashlib
import os

from jinja2 import Environment
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# copied verbatim from inspect_evals/gsm8k/gsm8k.py (installed copy)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
END_OF_TURN = "<end_of_turn>"


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _env() -> Environment:
    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg: str):
        raise TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env


with open(TEMPLATE_PATH, "r") as _f:
    _TEMPLATE = _env().from_string(_f.read())


def render(messages, add_generation_prompt: bool = True) -> str:
    return _TEMPLATE.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=BOS,
    )


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """inspect_evals.gsm8k.sample_to_fewshot"""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def build_prompt(question: str, fewshots: list[str] | None = None) -> str:
    """Render exactly what the grader sends to vLLM, up to <start_of_turn>model\\n."""
    messages = []
    if fewshots:
        messages.append({"role": "system", "content": "\n\n".join(fewshots)})
    messages.append({"role": "user", "content": user_prompt(question)})
    return render(messages, add_generation_prompt=True)


def build_completion(solution: str) -> str:
    """The assistant turn as the template would render it."""
    return solution.strip() + END_OF_TURN
