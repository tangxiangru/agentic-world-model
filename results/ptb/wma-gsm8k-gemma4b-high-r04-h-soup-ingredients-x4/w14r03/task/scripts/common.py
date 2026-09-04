"""Shared helpers: reproduce the grader's prompt byte-for-byte.

Everything here mirrors:
  /usr/local/lib/python3.10/dist-packages/inspect_evals/gsm8k/gsm8k.py
  /home/ben/task/templates/gemma3.jinja   (passed to vLLM by evaluate.py)
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
BASE_MODEL = (
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


def template_sha256() -> str:
    return hashlib.sha256(template_text().encode()).hexdigest()


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_system_message(fewshot: int = 10, fewshot_seed: int = 42) -> str:
    """Rebuild the exact 10-shot system message the grader prepends.

    hf_dataset(..., shuffle=True, seed=42, limit=10) -> datasets.shuffle(seed=42)
    then take the first `limit` rows.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=fewshot_seed)
    shots = []
    for rec in ds.select(range(fewshot)):
        answer = rec["answer"].split("####")
        target = answer.pop().strip()
        reasoning = "####".join(answer).strip()
        shots.append(f"{rec['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return "\n\n".join(shots)


def get_tokenizer(path: str = BASE_MODEL):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.chat_template = template_text()
    return tok


def render_prompt(tok, question: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_prompt(question)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render_target(completion: str) -> str:
    """What the jinja template emits for an assistant turn, minus the trailing \\n.

    The corpus stores the terminator explicitly so preflight's
    stop_token_consistent check can read it; strip it before applying `trim`
    so the result is byte-identical to what the template would produce.
    """
    body = completion.strip()
    if body.endswith(STOP_TOKEN):
        body = body[: -len(STOP_TOKEN)].strip()
    return body + STOP_TOKEN
