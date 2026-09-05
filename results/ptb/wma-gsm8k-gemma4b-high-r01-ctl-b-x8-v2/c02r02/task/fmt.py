"""Formatting helpers shared by data building, training and probing.

Everything here is derived from the grader itself
(inspect_evals/gsm8k/gsm8k.py + templates/gemma3.jinja) so that the string the
trainer sees is byte-for-byte the string vLLM will see at grading time.
"""
from __future__ import annotations

import functools
import hashlib
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_text() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha() -> str:
    return hashlib.sha256(template_text().encode()).hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def fewshot_system() -> str:
    """The exact 10-shot system message the grader builds (fewshot_seed=42)."""
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


def user_message(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


@functools.lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)
    tok.chat_template = template_text()
    return tok


def render_prompt(question: str, system: str | None = None) -> str:
    """Render exactly what vLLM receives, up to and including '<start_of_turn>model\\n'."""
    tok = _tokenizer()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_message(question)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render_example(question: str, solution: str, system: str | None = None) -> tuple[str, str]:
    """(prompt, completion) where completion ends with the grader's stop token."""
    return render_prompt(question, system), solution.strip() + "\n" + STOP_TOKEN
