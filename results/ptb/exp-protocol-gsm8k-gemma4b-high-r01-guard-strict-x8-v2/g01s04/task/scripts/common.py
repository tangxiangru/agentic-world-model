"""Shared pieces: the grader's exact prompt, template, and answer matching.

Everything here is copied from the harness so training and grading render the
same strings (pitfall: template_unreachable).
"""
from __future__ import annotations

import hashlib
import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"
END_OF_TURN_ID = 106


def template_text() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha() -> str:
    return hashlib.sha256(template_text().encode()).hexdigest()[:12]


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = template_text()
    return tok


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def sample_to_fewshot(question: str, reasoning: str, answer: str) -> str:
    """Verbatim shape of inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def build_messages(question: str, fewshots: list[str] | None = None):
    msgs = []
    if fewshots:
        msgs.append({"role": "system", "content": "\n\n".join(fewshots)})
    msgs.append({"role": "user", "content": user_prompt(question)})
    return msgs


# ---- grader-equivalent answer matching (inspect_ai.scorer._common.match_str,
# ---- location="end", numeric=True) ---------------------------------------
_NUM_PUNCT = re.compile(r"[,$%\s]")


def _normalize_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    # a token like "1e999" parses to inf and int(inf) raises OverflowError
    if v != v or v in (float("inf"), float("-inf")):
        return None
    if v == int(v):
        return str(int(v))
    return str(v)


def last_number(text: str) -> str | None:
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w2 = re.sub(r"[^0-9eE.\-+/]", "", w)
        n = _normalize_number(w2)
        if n is not None:
            return n
    return None


def is_correct(completion: str, target: str) -> bool:
    a = last_number(completion)
    t = _normalize_number(target)
    if a is None or t is None:
        return False
    return a == t
