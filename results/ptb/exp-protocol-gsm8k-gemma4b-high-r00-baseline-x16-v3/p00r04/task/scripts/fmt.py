"""Shared prompt/target formatting, byte-identical to what the grader renders.

The grader (evaluate.py) hands vLLM `templates/gemma3.jinja` as the chat template
and runs `inspect_evals/gsm8k` with its default 10-shot system message.  Everything
here is derived from those two files so training and grading cannot drift
(pitfall: template_unreachable).
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
FEWSHOT_PATH = os.path.join(TASK_DIR, "data", "fewshot_system.txt")
SNAPSHOT = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)

# copied verbatim from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
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


def template_sha() -> str:
    return hashlib.sha256(template_text().encode()).hexdigest()[:12]


def fewshot_system() -> str:
    with open(FEWSHOT_PATH) as f:
        return f.read()


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(tok, question: str, fewshot: bool) -> str:
    """The exact string vLLM feeds the model, up to and including
    '<start_of_turn>model\\n'."""
    messages = []
    if fewshot:
        messages.append({"role": "system", "content": fewshot_system()})
    messages.append({"role": "user", "content": user_content(question)})
    return tok.apply_chat_template(
        messages,
        chat_template=template_text(),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_target(solution: str, answer: str) -> str:
    """Assistant turn: reasoning, then the single ANSWER line, then the stop token."""
    body = solution.strip()
    return f"{body}\n\n{ANSWER_MARKER}{answer}{STOP_TOKEN}"
