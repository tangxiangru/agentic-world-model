"""Prompt/target rendering that is byte-identical to what the grader builds.

The grader is inspect_evals/gsm8k:
  * system_message(...)   -> 10 few-shot examples joined by "\n\n"   (only when fewshot>0)
  * prompt_template(MATH_PROMPT_TEMPLATE) wraps the question
  * the chat string is rendered by templates/gemma3.jinja (passed to vLLM as --chat-template)
  * scorer = match(numeric=True, location="end") -> reads the LAST numeric whitespace token

Everything here is derived from the gsm8k TRAIN split only. The test copy is never read.
"""

from __future__ import annotations

import hashlib
import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# byte-for-byte copy of inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_text() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha() -> str:
    return hashlib.sha256(template_text().encode()).hexdigest()[:12]


def user_message(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question.strip())


def build_messages(question: str, system: str | None = None) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_message(question)})
    return msgs


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """inspect_evals.gsm8k.sample_to_fewshot, verbatim."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def render_prompt(tokenizer, question: str, system: str | None = None) -> str:
    """The exact string vLLM sees before the model's first generated token."""
    return tokenizer.apply_chat_template(
        build_messages(question, system),
        chat_template=template_text(),
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------- target text

_ANSWER_LINE_RE = re.compile(r"^ANSWER:\s*.*$", re.MULTILINE)


def normalise_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_target(reasoning: str, answer: str) -> str:
    """One reasoning body, exactly one 'ANSWER: N' line, nothing after it."""
    body = reasoning.strip()
    # kill any other answer marker the source dataset may carry
    body = _ANSWER_LINE_RE.sub("", body).strip()
    return f"{body}\n\nANSWER: {normalise_number(answer)}"


def last_numeric_token(text: str) -> str | None:
    """Reimplementation of inspect's match(numeric=True, location='end') extraction."""
    v = text.strip().replace(",", "").replace("$", "")
    words = re.split(r"\s+", v)
    words.reverse()
    for w in words:
        w = w.strip().rstrip(".").rstrip("*").strip()
        if w.replace(".", "").isnumeric():
            return w
    return None


def graded_correct(completion: str, gold: str) -> bool:
    got = last_numeric_token(completion)
    if got is None:
        return False
    try:
        return abs(float(got) - float(normalise_number(gold))) < 1e-6
    except ValueError:
        return got == normalise_number(gold)
