"""Shared pieces: the grader's exact prompt/template, reproduced byte-for-byte.

Sources (read, not guessed):
  /usr/local/lib/python3.10/dist-packages/inspect_evals/gsm8k/gsm8k.py
  /home/ben/task/templates/gemma3.jinja
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
GSM8K_SNAPSHOT = (
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866"
)

# inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE, verbatim
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
START = "<start_of_turn>"
END = "<end_of_turn>"


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fewshot_prefix() -> str:
    """The system message inspect builds: 10 shuffled train items, seed 42."""
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
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def render_prompt(question: str, system: str | None = None) -> str:
    """Render exactly what templates/gemma3.jinja produces for
    [system?, user] with add_generation_prompt=True."""
    prefix = (system.strip() + "\n\n") if system else ""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    return f"{BOS}{START}user\n{prefix}{user}{END}\n{START}model\n"


def render_target(solution: str) -> str:
    """Assistant turn body + the terminator the grader stops on."""
    return solution.strip() + END + "\n"
