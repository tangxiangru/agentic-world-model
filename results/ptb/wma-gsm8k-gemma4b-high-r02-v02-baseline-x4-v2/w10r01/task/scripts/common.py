"""Shared prompt/format helpers.

Everything here mirrors what the grader actually does, so that training and
grading render the same strings:

  * the user turn is inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE, verbatim
  * the conversation is rendered with templates/gemma3.jinja, verbatim
  * the turn terminator is <end_of_turn> (id 106), which is in the base
    checkpoint's generation_config eos_token_id list
  * the grader is match(numeric=True, location="end"): the LAST number in the
    completion is what gets scored, so the target must end with the answer
"""
from __future__ import annotations

import re
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

# byte-for-byte copy of inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def user_turn(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question.strip())


def render_prompt(question: str, system: str | None = None) -> str:
    """Exactly what templates/gemma3.jinja produces with add_generation_prompt."""
    user = user_turn(question)
    if system:
        user = system.strip() + "\n\n" + user
    return f"{BOS}{SOT}user\n{user.strip()}{EOT}\n{SOT}model\n"


def render_target(solution: str, answer: str) -> str:
    """Body + the single answer marker + the turn terminator."""
    return f"{solution.strip()}\n\n{ANSWER_MARKER}{answer.strip()}{EOT}"


_CALC = re.compile(r"<<[^>]*>>")


def clean_gsm8k_reasoning(answer_field: str) -> str:
    body = answer_field.split("####")[0]
    return _CALC.sub("", body).strip()


def norm_answer(a: str) -> str:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if a.endswith(".0"):
        a = a[:-2]
    return a


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(completion: str) -> str | None:
    """Reproduce match(numeric=True, location='end'): last number in the text."""
    text = completion.strip()
    idx = text.rfind(ANSWER_MARKER)
    if idx >= 0:
        text = text[idx + len(ANSWER_MARKER):]
    nums = _NUM.findall(text)
    if not nums:
        return None
    return norm_answer(nums[-1])


def graded_correct(completion: str, gold: str) -> bool:
    got = extract_answer(completion)
    if got is None:
        return False
    try:
        return abs(float(got) - float(norm_answer(gold))) < 1e-6
    except ValueError:
        return got == norm_answer(gold)
