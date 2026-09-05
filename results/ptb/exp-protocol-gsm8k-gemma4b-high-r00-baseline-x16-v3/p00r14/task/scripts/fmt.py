"""Rendering that is byte-for-byte what the grader does.

The grader (evaluate.py -> inspect_evals/gsm8k) builds two messages, a 10-shot
system message and a user message from MATH_PROMPT_TEMPLATE, and renders them
with templates/gemma3.jinja. Training has to see the same string, so both the
template and the prompt template are read from the same files the grader reads
rather than copied by hand.
"""
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE, imported so it cannot drift.
from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE  # noqa: E402

STOP = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def chat_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.replace("{prompt}", question)


def render_prompt(tokenizer, question: str, system: str | None = None) -> str:
    """The exact prefix vLLM will be given at grading time."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content(question)})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=chat_template(),
    )


def render_target(solution: str) -> str:
    """What the model must produce: the solution, then the terminator."""
    return solution.strip() + STOP
