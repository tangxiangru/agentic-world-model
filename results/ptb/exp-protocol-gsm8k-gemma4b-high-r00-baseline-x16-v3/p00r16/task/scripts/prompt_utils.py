"""Exact replication of the harness prompt/grading pipeline.

Everything here mirrors inspect_evals.gsm8k + templates/gemma3.jinja so that
training data, local probes and the official evaluate.py all render the same
string. Do not "improve" these strings; they are byte-for-byte copies.
"""
from __future__ import annotations

import functools
import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# copied from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


@functools.lru_cache(maxsize=1)
def template_text() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


@functools.lru_cache(maxsize=1)
def template_sha() -> str:
    return hashlib.sha256(template_text().encode()).hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def fewshot_system_message() -> str:
    """The exact 10-shot system message inspect builds (fewshot_seed=42)."""
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


def render(tokenizer, messages, add_generation_prompt=True) -> str:
    return tokenizer.apply_chat_template(
        messages,
        chat_template=template_text(),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def eval_prompt(tokenizer, question: str, fewshot: bool = True) -> str:
    msgs = []
    if fewshot:
        msgs.append({"role": "system", "content": fewshot_system_message()})
    msgs.append({"role": "user", "content": user_prompt(question)})
    return render(tokenizer, msgs)


def grade(completion: str, gold: str) -> bool:
    """Identical to inspect_ai match(numeric=True, location='end')."""
    from inspect_ai.scorer._common import match_str

    _, ok = match_str(value=completion, target=str(gold), location="end",
                      ignore_case=True, numeric=True)
    return bool(ok)
