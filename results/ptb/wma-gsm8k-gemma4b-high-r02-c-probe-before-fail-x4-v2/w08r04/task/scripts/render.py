"""Prompt/target rendering that is byte-identical to what the grader produces.

Everything grader-side is imported from the installed inspect_evals package or
read from templates/gemma3.jinja, so training and grading cannot drift apart.
Guards against the `template_unreachable` pitfall.
"""
from __future__ import annotations

import hashlib
import os

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE  # noqa: F401

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# sha256 of templates/gemma3.jinja as shipped; recorded so a silent edit is caught.
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def load_template() -> str:
    with open(TEMPLATE_PATH, "r") as f:
        text = f.read()
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != TEMPLATE_SHA256:
        raise RuntimeError(f"templates/gemma3.jinja changed: {got} != {TEMPLATE_SHA256}")
    return text


def template_sha256() -> str:
    return hashlib.sha256(load_template().encode()).hexdigest()


def user_message(question: str) -> str:
    """The user turn the grader builds: prompt_template(MATH_PROMPT_TEMPLATE)."""
    return MATH_PROMPT_TEMPLATE.replace("{prompt}", question)


def fewshot_system_message(shots):
    """shots: list of (question, reasoning, answer). Mirrors sample_to_fewshot."""
    return "\n\n".join(
        f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}" for q, r, a in shots
    )


def render_prompt(tokenizer, question: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_message(question)})
    return tokenizer.apply_chat_template(
        msgs,
        chat_template=load_template(),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_completion(solution: str, answer: str) -> str:
    """Model turn exactly as the template would emit it: trimmed body + stop token."""
    body = f"{solution.strip()}\n\n{ANSWER_MARKER}{answer}".strip()
    return body + STOP_TOKEN
