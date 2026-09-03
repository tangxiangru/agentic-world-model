"""Shared formatting helpers: reproduce the grader's prompt byte-for-byte.

The grader (inspect_evals/gsm8k) builds, for every test item:
  system  : 10 few-shot exemplars from openai/gsm8k main/train, seed 42, shuffled
  user    : MATH_PROMPT_TEMPLATE.format(prompt=question)
and renders them with templates/gemma3.jinja (NOT the tokenizer's own template).
Training must render the same strings, so both sides live here.
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def load_template() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != TEMPLATE_SHA256:
        raise SystemExit(f"templates/gemma3.jinja changed: {got} != {TEMPLATE_SHA256}")
    return raw.decode("utf-8")


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_system_message() -> str:
    """The exact system message the grader uses (fewshot=10, seed=42, shuffled)."""
    from inspect_ai.dataset import Sample, hf_dataset

    def record_to_sample(record):
        DELIM = "####"
        answer = record["answer"].split(DELIM)
        target = answer.pop().strip()
        reasoning = DELIM.join(answer)
        return Sample(
            id=None, input=record["question"], target=target,
            metadata={"reasoning": reasoning.strip()},
        )

    fewshots = hf_dataset(
        path="openai/gsm8k", data_dir="main", split="train",
        sample_fields=record_to_sample, shuffle=True, seed=42, limit=10,
    )
    return "\n\n".join(
        f"{s.input}\n\nReasoning:\n{s.metadata['reasoning']}\n\nANSWER: {s.target}"
        for s in fewshots
    )
