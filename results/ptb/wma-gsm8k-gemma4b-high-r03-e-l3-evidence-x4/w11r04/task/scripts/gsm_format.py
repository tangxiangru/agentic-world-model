"""Shared formatting helpers: render exactly what the grader renders.

The grader is `inspect_evals/gsm8k` (fewshot=10, fewshot_seed=42, shuffle=True)
plus `templates/gemma3.jinja`.  Everything here is derived from those two files
so that a training row and an eval prompt are byte-identical strings.
"""
from __future__ import annotations

import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# copied byte-for-byte from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANSWER_MARKER = "ANSWER: "
STOP_TOKEN = "<end_of_turn>"


def chat_template() -> str:
    with open(CHAT_TEMPLATE_PATH) as f:
        return f.read()


def user_turn(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def sample_to_fewshot(question: str, reasoning: str, target: str) -> str:
    """inspect_evals/gsm8k.sample_to_fewshot, with the fields it reads."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def eval_fewshot_system() -> str:
    """The exact 10-shot system message the grader puts in front of every item."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    # inspect_ai's MemoryDataset.shuffle -> random.Random(seed).shuffle(indices)
    import random

    idx = list(range(len(ds)))
    random.Random(42).shuffle(idx)
    shots = []
    for i in idx[:10]:
        rec = ds[i]
        reasoning, target = split_gsm8k_answer(rec["answer"])
        shots.append(sample_to_fewshot(rec["question"], reasoning, target))
    return "\n\n".join(shots)


def split_gsm8k_answer(answer: str) -> tuple[str, str]:
    parts = answer.split("####")
    target = parts.pop().strip()
    reasoning = "####".join(parts).strip()
    return reasoning, target


_CALC = re.compile(r"<<[^>]*>>")


def strip_calculator(text: str) -> str:
    return _CALC.sub("", text)


def normalise_number(s: str) -> str:
    """Grader-side normalisation: match(numeric=True) strips $ , and % ."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if s.endswith("."):
        s = s[:-1]
    return s
