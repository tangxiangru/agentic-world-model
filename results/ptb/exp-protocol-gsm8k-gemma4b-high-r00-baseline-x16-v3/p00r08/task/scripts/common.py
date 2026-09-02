#!/usr/bin/env python3
"""Shared rendering helpers.

Everything the grader does to build a prompt is reproduced here, from the same
files the grader reads:
  * the prompt wrapper   -> inspect_evals.gsm8k.gsm8k.MATH_PROMPT_TEMPLATE
  * the few-shot prefix  -> inspect_evals.gsm8k.gsm8k.sample_to_fewshot
  * the chat template    -> /home/ben/task/templates/gemma3.jinja (byte-for-byte)
This is the `template_unreachable` pitfall: nothing here may be re-typed by hand.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "templates" / "gemma3.jinja"

STOP_TOKEN = "<end_of_turn>"
STOP_TOKEN_ID = 106
ANSWER_MARKER = "ANSWER: "


def chat_template() -> str:
    return TEMPLATE_PATH.read_text()


def template_hash() -> str:
    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()[:16]


@lru_cache(maxsize=1)
def prompt_wrapper() -> str:
    from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

    return MATH_PROMPT_TEMPLATE


def user_content(question: str) -> str:
    """Exactly what inspect's prompt_template() produces for one item."""
    return prompt_wrapper().replace("{prompt}", question)


@lru_cache(maxsize=1)
def eval_fewshot_samples():
    """The ten few-shot items the grader puts in its system message."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample

    return list(
        hf_dataset(
            path="openai/gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            shuffle=True,
            seed=42,
            limit=10,
        )
    )


def fewshot_block(sample) -> str:
    from inspect_evals.gsm8k.gsm8k import sample_to_fewshot

    return sample_to_fewshot(sample)


def system_message(samples) -> str:
    return "\n\n".join(fewshot_block(s) for s in samples)


def render_prompt(tokenizer, question: str, system: str | None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content(question)})
    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template(),
        tokenize=False,
        add_generation_prompt=True,
    )
