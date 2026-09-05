"""Prompt/target formatting shared by data building, training and probing.

Everything here mirrors the grading harness byte-for-byte:
  * MATH_PROMPT_TEMPLATE is copied from inspect_evals/gsm8k/gsm8k.py
  * few-shot system messages are built exactly like sample_to_fewshot()
  * the chat template is templates/gemma3.jinja, the same file evaluate.py
    hands to vLLM (pitfall: template_unreachable)
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def load_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """Exactly inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def build_messages(question: str, fewshots: list[tuple[str, str, str]] | None = None):
    msgs = []
    if fewshots:
        msgs.append(
            {
                "role": "system",
                "content": "\n\n".join(fewshot_block(*fs) for fs in fewshots),
            }
        )
    msgs.append({"role": "user", "content": user_content(question)})
    return msgs
