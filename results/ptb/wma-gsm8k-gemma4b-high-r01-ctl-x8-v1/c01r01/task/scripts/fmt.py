"""Exact reproduction of the grader's prompt/target surface.

Everything here is derived from two files that must not be modified:
  /usr/local/lib/python3.10/dist-packages/inspect_evals/gsm8k/gsm8k.py
  /home/ben/task/templates/gemma3.jinja
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

# inspect_evals/gsm8k/gsm8k.py :: MATH_PROMPT_TEMPLATE
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


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """inspect_evals/gsm8k/gsm8k.py :: sample_to_fewshot"""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def load_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha() -> str:
    return hashlib.sha256(load_template().encode()).hexdigest()[:16]


def official_fewshot_system(n: int = 10, seed: int = 42) -> str:
    """Reproduce the exact system message the grader builds.

    inspect_ai.dataset.hf_dataset(path, data_dir='main', split='train',
        shuffle=True, seed=42, limit=10) -> datasets .shuffle(seed).select(range(limit))
    then '\\n\\n'.join(sample_to_fewshot(s) for s in fewshots)
    """
    from datasets import load_dataset

    ds = load_dataset(
        "openai/gsm8k", "main", split="train",
        revision="740312add88f781978c0658806c59bc2815b9866",
    )
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(n))
    blocks = []
    for rec in ds:
        q = rec["question"]
        parts = rec["answer"].split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        blocks.append(fewshot_block(q, reasoning, target))
    return "\n\n".join(blocks)


def render(tokenizer, system: str | None, question: str, completion: str | None = None):
    """Render exactly as the grader does.

    Returns (prompt_text, full_text). full_text == prompt_text + completion + STOP_TOKEN.
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_prompt(question)})
    prompt_text = tokenizer.apply_chat_template(
        msgs, chat_template=load_template(), tokenize=False, add_generation_prompt=True
    )
    if completion is None:
        return prompt_text, None
    return prompt_text, prompt_text + completion + STOP_TOKEN
