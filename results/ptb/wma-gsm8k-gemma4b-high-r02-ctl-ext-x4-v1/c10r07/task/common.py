"""Shared, single source of truth for how a GSM8K prompt is rendered.

Everything here mirrors the grader byte-for-byte:
  * MATH_PROMPT_TEMPLATE and the 10-shot system message come from the installed
    inspect_evals.gsm8k module itself (not a copy), so they cannot drift.
  * The chat template is read from templates/gemma3.jinja -- the same file
    evaluate.py hands to vLLM -- and hashed, so a silent edit is visible.
Pitfall guarded: template_unreachable, double_answer_format.
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

# the terminator the grading template writes after every turn; token id 106 and
# already present in the snapshot's generation_config eos_token_id [1, 106].
STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def chat_template() -> str:
    with open(TEMPLATE_PATH, "r") as f:
        return f.read()


def chat_template_sha() -> str:
    return hashlib.sha256(chat_template().encode()).hexdigest()[:12]


def math_prompt_template() -> str:
    from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

    return MATH_PROMPT_TEMPLATE


def user_message(question: str) -> str:
    return math_prompt_template().replace("{prompt}", question)


_FEWSHOT_CACHE: str | None = None


def fewshot_system_message() -> str:
    """The exact system message the grader builds: 10 gsm8k TRAIN items, seed 42."""
    global _FEWSHOT_CACHE
    if _FEWSHOT_CACHE is None:
        from inspect_ai.dataset import hf_dataset
        from inspect_evals.gsm8k.gsm8k import (
            DATASET_PATH,
            record_to_sample,
            sample_to_fewshot,
        )

        shots = hf_dataset(
            path=DATASET_PATH,
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            shuffle=True,
            seed=42,
            limit=10,
        )
        _FEWSHOT_CACHE = "\n\n".join(sample_to_fewshot(s) for s in shots)
    return _FEWSHOT_CACHE


def render_prompt(tokenizer, question: str, with_fewshot: bool) -> str:
    """Render the prompt string exactly as evaluate.py -> vLLM will render it."""
    messages = []
    if with_fewshot:
        messages.append({"role": "system", "content": fewshot_system_message()})
    messages.append({"role": "user", "content": user_message(question)})
    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template(),
        tokenize=False,
        add_generation_prompt=True,
    )


def format_target(solution_body: str, answer: str) -> str:
    """One answer marker, at the very end, then the stop token.

    The grader is match(numeric=True, location="end"): it reads the LAST
    whitespace-delimited numeric token of the completion. So nothing numeric may
    follow the answer.
    """
    body = solution_body.strip()
    return f"{body}\n\n{ANSWER_MARKER}{answer}{STOP_TOKEN}"
