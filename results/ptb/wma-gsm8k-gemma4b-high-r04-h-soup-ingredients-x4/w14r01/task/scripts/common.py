"""Shared prompt/format helpers.

Everything here mirrors, byte for byte, what the grader does:
  * the user-turn text comes from inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE
  * the few-shot block comes from inspect_evals.gsm8k.sample_to_fewshot
  * the chat rendering comes from templates/gemma3.jinja (NOT the tokenizer's own)
"""
from __future__ import annotations

import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANSWER_MARKER = "ANSWER: "
STOP_TOKEN = "<end_of_turn>"


def user_text(question: str, fewshot_block: str | None = None) -> str:
    """The full text of the single user turn the grader sends.

    gemma3.jinja folds the system message into the first user turn as
    `system_content + "\\n\\n"`, so a few-shot prefix is just prepended here.
    """
    body = MATH_PROMPT_TEMPLATE.format(prompt=question)
    if fewshot_block:
        return fewshot_block + "\n\n" + body
    return body


def sample_to_fewshot(question: str, reasoning: str, target: str) -> str:
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def render_prompt(user: str) -> str:
    """gemma3.jinja with a single user turn + generation prompt."""
    return f"<bos><start_of_turn>user\n{user.strip()}<end_of_turn>\n<start_of_turn>model\n"


def render_target(completion: str) -> str:
    """Idempotent: the jsonl targets already carry the stop token so that
    preflight's stop_token_consistent check can read it off the file."""
    c = completion.strip()
    return c if c.endswith(STOP_TOKEN) else c + STOP_TOKEN


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_number(s: str) -> str | None:
    """Return a canonical numeric string, or None if `s` is not a plain number."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
        if s in ("", "-"):
            s = "0"
    try:
        f = float(s)
    except ValueError:
        return None
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return s


def strip_calc_annotations(text: str) -> str:
    return re.sub(r"<<[^>]*>>", "", text)
