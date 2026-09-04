"""Exact reproduction of the grading prompt (inspect_evals/gsm8k + templates/gemma3.jinja).

Everything that shapes a training row lives here so training and grading cannot drift.
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
START = "<start_of_turn>"
END = "<end_of_turn>"
STOP_TOKEN = END


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """inspect_evals.gsm8k.sample_to_fewshot"""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def render(question: str, system: str | None = None, target: str | None = None) -> str:
    """Render the gemma3.jinja chat template by hand for the single-user-turn case.

    Matches templates/gemma3.jinja: the system message becomes a prefix inside the
    first user turn, contents are trimmed, turns end with <end_of_turn>\\n.
    """
    prefix = (system.strip() + "\n\n") if system else ""
    out = BOS
    out += f"{START}user\n{prefix}{user_prompt(question).strip()}{END}\n"
    out += f"{START}model\n"
    if target is not None:
        out += f"{target.strip()}{END}\n"
    return out


def target_text(reasoning: str, answer: str) -> str:
    """The assistant turn the grader wants: reasoning, then the final ANSWER line."""
    return f"{reasoning.strip()}\n\nANSWER: {answer.strip()}"
