"""Shared rendering helpers: the grader's exact prompt, byte-for-byte.

Everything here is derived from the two files the grader actually uses:
  /home/ben/task/templates/gemma3.jinja           (chat template passed to vLLM)
  inspect_evals/gsm8k/gsm8k.py                    (prompt template + few-shot builder)
so training and grading render the same string (pitfalls.yaml template_unreachable).
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
SNAPSHOT = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
            "cc012e0a6d0787b4adcc0fa2c4da74402494554d")

# copied verbatim from inspect_evals.gsm8k
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
    return hashlib.sha256(template_text().encode()).hexdigest()[:16]


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = template_text()
    return tok


def grader_fewshot_system() -> str:
    """The exact 10-shot system message inspect_evals builds (fewshot_seed=42).

    Reads the gsm8k TRAIN split only.
    """
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot
    fewshots = hf_dataset(
        path="openai/gsm8k", data_dir="main", split="train",
        sample_fields=record_to_sample, shuffle=True, seed=42, limit=10,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(tok, question: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_content(question)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render_completion(solution: str, answer: str) -> str:
    return solution.rstrip() + "\n\n" + ANSWER_MARKER + answer + STOP_TOKEN
