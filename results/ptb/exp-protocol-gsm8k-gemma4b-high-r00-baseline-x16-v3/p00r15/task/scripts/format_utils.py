#!/usr/bin/env python3
"""Rendering helpers shared by the trainer and the pre-flight checks.

Everything here exists to make training render byte-for-byte the same string the
grader renders (pitfall: template_unreachable). The chat template is read from
templates/gemma3.jinja -- the same file evaluate.py hands to vLLM -- and hashed.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
from pathlib import Path

TASK_DIR = Path("/home/ben/task")
TEMPLATE_PATH = TASK_DIR / "templates" / "gemma3.jinja"
GSM8K_TRAIN = TASK_DIR / "data" / "gsm8k_train_raw.jsonl"

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"


def template_text() -> str:
    return TEMPLATE_PATH.read_text()


def template_sha() -> str:
    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()[:12]


def render_prompt(tokenizer, user_content: str, system_content: str | None = None) -> str:
    """Render exactly what vLLM will be given at generation time."""
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        messages,
        chat_template=template_text(),
        tokenize=False,
        add_generation_prompt=True,
    )


@functools.lru_cache(maxsize=1)
def _gsm8k_train_rows() -> list[dict]:
    rows = []
    with open(GSM8K_TRAIN) as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _to_fewshot(rec: dict) -> str:
    """inspect_evals' sample_to_fewshot, applied to a raw gsm8k train record."""
    body, _, target = rec["answer"].rpartition("####")
    return f"{rec['question']}\n\nReasoning:\n{body.strip()}\n\nANSWER: {target.strip()}"


def eval_fewshot_system(n: int = 10, seed: int = 42) -> str:
    """The system message the grader builds: hf_dataset(train, shuffle=True, seed=42)[:n]."""
    from inspect_ai.dataset import hf_dataset

    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    ds = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in ds)


def random_fewshot_system(rng: random.Random, k: int) -> str:
    """A k-shot system message in the grader's format, drawn from the gsm8k TRAIN split."""
    rows = _gsm8k_train_rows()
    picks = rng.sample(rows, k)
    return "\n\n".join(_to_fewshot(r) for r in picks)
