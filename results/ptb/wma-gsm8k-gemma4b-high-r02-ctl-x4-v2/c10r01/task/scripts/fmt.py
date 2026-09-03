"""Rendering helpers shared by data building, training and dry runs.

Everything here reproduces, byte for byte, what the grader (evaluate.py ->
inspect_evals/gsm8k -> vLLM OpenAI server with --chat-template
templates/gemma3.jinja) puts in front of the model.
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# inspect_evals.gsm8k.gsm8k.MATH_PROMPT_TEMPLATE, copied verbatim
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


def sample_to_fewshot(question: str, reasoning: str, answer: str) -> str:
    """inspect_evals.gsm8k.gsm8k.sample_to_fewshot, copied verbatim."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def user_message(question: str, system: str | None) -> str:
    """The single user turn the gemma3 template renders.

    gemma3.jinja folds the system message into the first user turn as
    `system + '\\n\\n' + user`, then `| trim`s the user content.
    """
    body = MATH_PROMPT_TEMPLATE.format(prompt=question)
    if system:
        return system + "\n\n" + body
    return body


def render_prompt(question: str, system: str | None = None) -> str:
    """Exactly what the model is conditioned on, including <bos>."""
    return (
        "<bos><start_of_turn>user\n"
        + user_message(question, system).strip()
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def render_completion(solution: str, answer: str) -> str:
    """Target text: reasoning, one ANSWER line, then the stop token."""
    return solution.strip() + "\n\n" + ANSWER_MARKER + str(answer).strip() + STOP_TOKEN
