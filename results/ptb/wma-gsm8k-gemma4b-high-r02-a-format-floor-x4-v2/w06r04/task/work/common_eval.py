"""Shared rendering + scoring helpers that mirror the official grader exactly.

The official harness is inspect_evals/gsm8k served through vLLM with
templates/gemma3.jinja:
  * system message = 10 few-shot examples from the gsm8k TRAIN split
    (hf_dataset(..., shuffle=True, seed=42, limit=10)), joined by "\n\n",
    each rendered as  "{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"
  * user message   = MATH_PROMPT_TEMPLATE.format(prompt=question)
  * scorer         = match(location="end", numeric=True)

Everything here is imported from the installed inspect packages so it cannot
drift from what the grader does.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

from inspect_evals.gsm8k.gsm8k import (  # noqa: E402
    MATH_PROMPT_TEMPLATE,
    record_to_sample,
    sample_to_fewshot,
)
from inspect_ai.dataset import hf_dataset  # noqa: E402
from inspect_ai.scorer._common import match_str  # noqa: E402

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMMA_TEMPLATE = os.path.join(TASK_DIR, "templates", "gemma3.jinja")


def fewshot_system_message(fewshot: int = 10, fewshot_seed: int = 42) -> str:
    """Byte-identical to what inspect_evals/gsm8k builds."""
    shots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=fewshot_seed,
        limit=fewshot,
    )
    return "\n\n".join([sample_to_fewshot(s) for s in shots])


def user_message(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def load_template() -> str:
    with open(GEMMA_TEMPLATE) as f:
        return f.read()


def render(messages, add_generation_prompt=True, bos_token="<bos>"):
    """Render with the grader's own jinja template."""
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError

    def raise_exception(msg):
        raise TemplateError(msg)

    env = Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = raise_exception
    tmpl = env.from_string(load_template())
    return tmpl.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=bos_token,
    )


def grade(completion: str, target: str) -> bool:
    _, ok = match_str(
        value=completion, target=str(target), location="end",
        ignore_case=True, numeric=True,
    )
    return bool(ok)
