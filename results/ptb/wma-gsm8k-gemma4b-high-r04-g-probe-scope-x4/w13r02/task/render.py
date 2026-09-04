"""Shared rendering: the grader's chat template, reproduced byte-for-byte.

Every training row must be rendered with the *same* template the grader passes
to vLLM (templates/gemma3.jinja). This module renders with jinja from that file
so training and grading cannot drift (pitfalls.yaml: template_unreachable).
"""
from __future__ import annotations

import hashlib
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "gemma3.jinja")

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

START = "<start_of_turn>"
END = "<end_of_turn>"
BOS = "<bos>"


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def render_prompt(question: str, system: str | None = None) -> str:
    """Prompt string up to and including the model generation prompt."""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question.strip()).strip()
    prefix = (system.strip() + "\n\n") if system else ""
    return f"{BOS}{START}user\n{prefix}{user}{END}\n{START}model\n"


def render_target(completion: str) -> str:
    """Assistant turn as the grader's template would close it.

    Data files already carry the stop token so preflight can verify them; do not
    append a second one.
    """
    c = completion.strip()
    return c if c.endswith(END) else c + END


def jinja_render(question: str, system: str | None = None) -> str:
    """Same thing, but through the actual jinja file, for verification."""
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError

    def raise_exception(msg: str):
        raise TemplateError(msg)

    env = Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = raise_exception
    with open(TEMPLATE_PATH) as f:
        tmpl = env.from_string(f.read())
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question.strip())}
    )
    return tmpl.render(messages=messages, bos_token=BOS, add_generation_prompt=True)


def build_fewshot_shots() -> list[str]:
    """The ten shot blocks inspect_evals builds, in its order (gsm8k train, seed 42)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42).select(range(10))
    parts = []
    for r in ds:
        answer = r["answer"].split("####")
        target = answer.pop().strip()
        reasoning = "####".join(answer).strip()
        parts.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return parts


def build_fewshot_system_text() -> str:
    """The exact 10-shot system message inspect_evals builds (gsm8k train, seed 42)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42).select(range(10))
    parts = []
    for r in ds:
        answer = r["answer"].split("####")
        target = answer.pop().strip()
        reasoning = "####".join(answer).strip()
        parts.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return "\n\n".join(parts)
