"""Shared prompt utilities that mirror the inspect_evals gsm8k task exactly."""
from __future__ import annotations

import os

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

_FEWSHOT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fewshot_system.txt")


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_system_message() -> str:
    """Reproduce the 10-shot system message used by inspect_evals/gsm8k (seed 42)."""
    if os.path.exists(_FEWSHOT_CACHE):
        with open(_FEWSHOT_CACHE) as f:
            return f.read()
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    msg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])
    os.makedirs(os.path.dirname(_FEWSHOT_CACHE), exist_ok=True)
    with open(_FEWSHOT_CACHE, "w") as f:
        f.write(msg)
    return msg


def build_messages(question: str, with_fewshot: bool) -> list[dict]:
    msgs = []
    if with_fewshot:
        msgs.append({"role": "system", "content": fewshot_system_message()})
    msgs.append({"role": "user", "content": user_prompt(question)})
    return msgs
