#!/usr/bin/env python3
"""The grader's prompt format, reproduced byte-for-byte.

Everything here is read from the same places the grader reads it from:
  * MATH_PROMPT_TEMPLATE and sample_to_fewshot from inspect_evals.gsm8k.gsm8k
  * the chat template from /home/ben/task/templates/gemma3.jinja (the file
    evaluate.py passes to vLLM), never from the tokenizer, which ships none.

pitfalls.yaml:template_unreachable is the reason this module exists: training
must render the exact string the grader renders.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE, sample_to_fewshot

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


@lru_cache(maxsize=1)
def chat_template() -> str:
    with open(TEMPLATE_PATH) as f:
        src = f.read()
    got = hashlib.sha256(src.encode()).hexdigest()
    if TEMPLATE_SHA256 and got != TEMPLATE_SHA256:
        raise RuntimeError(f"gemma3.jinja changed: {got} != {TEMPLATE_SHA256}")
    return src


def user_message(question: str) -> str:
    """Exactly what prompt_template(MATH_PROMPT_TEMPLATE) produces."""
    return MATH_PROMPT_TEMPLATE.replace("{prompt}", question)


def fewshot_block(examples: list[tuple[str, str, str]]) -> str:
    """The system message: '\\n\\n'.join(sample_to_fewshot(s) for s in shots).

    examples: (question, reasoning, answer) triples, reasoning being the gsm8k
    train solution body with the '#### N' line removed.
    """
    class _S:  # duck-types the fields sample_to_fewshot touches
        def __init__(self, q, r, a):
            self.input, self.metadata, self.target = q, {"reasoning": r}, a

    return "\n\n".join(sample_to_fewshot(_S(q, r, a)) for q, r, a in examples)


def render(tokenizer, question: str, system: str | None, completion: str | None):
    """Render prompt (and optionally prompt+completion) with the grader's template.

    Returns (prompt_text, full_text). full_text is None when completion is None.
    The assistant turn is closed with <end_of_turn>\\n exactly as the template
    does, so the trained terminator is the one vLLM stops on.
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_message(question)})
    prompt = tokenizer.apply_chat_template(
        msgs, chat_template=chat_template(), tokenize=False, add_generation_prompt=True
    )
    if completion is None:
        return prompt, None
    full = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": completion}],
        chat_template=chat_template(), tokenize=False, add_generation_prompt=False,
    )
    return prompt, full
