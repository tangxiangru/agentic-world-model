"""Shared helpers: render exactly what the grader renders.

The grader (evaluate.py -> inspect_evals/gsm8k) does three things we must mirror
byte-for-byte in training data:

  1. wraps the question in MATH_PROMPT_TEMPLATE,
  2. renders the conversation with templates/gemma3.jinja (NOT the tokenizer's
     own chat template -- see pitfall `template_unreachable`),
  3. scores with match(numeric=True, location="end"), which walks the completion
     backwards and takes the first whitespace-token that is numeric after
     punctuation stripping.

So a training target must (a) end with <end_of_turn>, which is in the base
model's generation_config eos_token_id list and therefore a vLLM stop token,
and (b) have the final answer as the last number in the text.
"""
from __future__ import annotations

import hashlib
import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# byte-for-byte copy of inspect_evals.gsm8k.gsm8k.MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"
BOS = "<bos>"


def template_sha() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def assert_template_matches_grader() -> None:
    """Fail loudly if our copy of the prompt template drifts from the harness."""
    from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE as GRADER_TPL

    assert MATH_PROMPT_TEMPLATE == GRADER_TPL, "MATH_PROMPT_TEMPLATE drifted from inspect_evals"


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(question: str, system: str | None = None) -> str:
    """Exactly what templates/gemma3.jinja emits with add_generation_prompt=True."""
    u = user_content(question).strip()
    if system:
        u = system.strip() + "\n\n" + u
    return f"{BOS}<start_of_turn>user\n{u}{END_OF_TURN}\n<start_of_turn>model\n"


def render_completion(solution: str) -> str:
    """Assistant turn as the template would emit it (content is `| trim`ed)."""
    return f"{solution.strip()}{END_OF_TURN}\n"


def sample_to_fewshot(question: str, reasoning: str, answer: str) -> str:
    """Byte-for-byte copy of inspect_evals.gsm8k.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


_INT_RE = re.compile(r"^-?\d{1,12}$")


def is_plain_int(s: str) -> bool:
    return bool(_INT_RE.match((s or "").strip().replace(",", "")))


def last_number(text: str) -> str | None:
    """Reimplementation of inspect's grader read: last numeric whitespace-token."""
    from inspect_ai._util.text import strip_numeric_punctuation

    v = strip_numeric_punctuation(text.strip().casefold())
    words = re.split(r"\s+", v)
    words.reverse()
    for w in words:
        if w.replace(".", "").isnumeric():
            return w
    return None


def grader_reads(completion: str, target: str) -> bool:
    """True iff inspect's match(numeric=True) would score this completion correct."""
    from inspect_ai.scorer._common import match_str

    _, ok = match_str(value=completion, target=target, location="end", ignore_case=True, numeric=True)
    return ok
