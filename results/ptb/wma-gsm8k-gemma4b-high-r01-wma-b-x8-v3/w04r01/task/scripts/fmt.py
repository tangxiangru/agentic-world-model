"""Rendering that matches the grader byte-for-byte.

The grader (evaluate.py -> inspect_ai -> vllm) renders the conversation with
templates/gemma3.jinja and prompts with inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE.
Everything training-side goes through this module so the two cannot drift.
Pitfall: template_unreachable (pitfalls.yaml).
"""
from __future__ import annotations

import hashlib
import os

from jinja2 import Environment
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
BASE_MODEL = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"      # id 106, in generation_config.eos_token_id
ANSWER_MARKER = "ANSWER: "
BOS = "<bos>"


def template_text() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha() -> str:
    return hashlib.sha256(template_text().encode()).hexdigest()[:12]


def _env() -> Environment:
    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg: str):
        raise TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env


_TPL = None


def render_chat(messages, add_generation_prompt=True) -> str:
    """Render with the grader's own jinja file (not the tokenizer's template)."""
    global _TPL
    if _TPL is None:
        _TPL = _env().from_string(template_text())
    return _TPL.render(
        messages=messages, add_generation_prompt=add_generation_prompt, bos_token=BOS
    )


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(question: str, fewshot_system: str | None = None) -> str:
    msgs = []
    if fewshot_system:
        msgs.append({"role": "system", "content": fewshot_system})
    msgs.append({"role": "user", "content": user_content(question)})
    return render_chat(msgs, add_generation_prompt=True)


def render_completion(reasoning: str, answer: str) -> str:
    """Assistant turn: reasoning, then the single answer marker, then the stop token."""
    body = reasoning.strip()
    return f"{body}\n\n{ANSWER_MARKER}{answer}{STOP_TOKEN}"
