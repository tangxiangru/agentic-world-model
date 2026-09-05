"""The exact strings the grader uses, reproduced once so training and grading agree.

Everything here is read from the same sources the grader reads:
  * MATH_PROMPT_TEMPLATE and the 10-shot system message come from
    inspect_evals.gsm8k (the module evaluate.py imports).
  * The chat template comes from templates/gemma3.jinja (the file evaluate.py
    hands to vLLM), hash-checked so a silent edit cannot slip through.

pitfalls.yaml: template_unreachable, eos_mismatch, double_answer_format.
"""

from __future__ import annotations

import hashlib
import os

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import (
    DATASET_PATH,
    MATH_PROMPT_TEMPLATE,
    record_to_sample,
    sample_to_fewshot,
)

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# sha256 of templates/gemma3.jinja as evaluate.py ships it.
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

# The terminator the grading template emits after a model turn, and the last
# marker the grader's match(numeric=True, location="end") scorer can see.
STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def chat_template(check_hash: bool = True) -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if check_hash and TEMPLATE_SHA256 and digest != TEMPLATE_SHA256:
        raise RuntimeError(
            f"templates/gemma3.jinja changed: {digest} != {TEMPLATE_SHA256}"
        )
    return raw.decode("utf-8")


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fewshot_system_message(fewshot: int = 10, fewshot_seed: int = 42) -> str:
    """Byte-identical to the system message inspect_evals/gsm8k builds.

    Drawn from the gsm8k TRAIN split, exactly as the task does.
    """
    shots = hf_dataset(
        path=DATASET_PATH,
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=fewshot_seed,
        limit=fewshot,
    )
    return "\n\n".join([sample_to_fewshot(s) for s in shots])


def user_prompt(question: str) -> str:
    """The user turn the grader sends: MATH_PROMPT_TEMPLATE with {prompt} filled."""
    return MATH_PROMPT_TEMPLATE.replace("{prompt}", question)


def messages(question: str, system: str | None = None) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_prompt(question)})
    return msgs
