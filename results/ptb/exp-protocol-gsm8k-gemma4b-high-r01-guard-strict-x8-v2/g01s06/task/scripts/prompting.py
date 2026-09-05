"""Prompt rendering that reproduces the grader byte-for-byte.

The grader is inspect_evals/gsm8k with fewshot=10, rendered through
templates/gemma3.jinja by vLLM's OpenAI chat endpoint.  Everything here is
copied from those two files; `render_prompt` must produce exactly what the
server produces for the same message list.
"""
from __future__ import annotations

# verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """inspect_evals.gsm8k.sample_to_fewshot"""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def render_prompt(question: str, system: str | None = None) -> str:
    """templates/gemma3.jinja with add_generation_prompt=True, one user turn."""
    first_user_prefix = (system + "\n\n") if system else ""
    return (
        "<bos>"
        "<start_of_turn>user\n"
        + first_user_prefix
        + user_content(question).strip()
        + "<end_of_turn>\n"
        + "<start_of_turn>model\n"
    )


def render_target(solution: str) -> str:
    """Assistant turn: content is `| trim`-ed by the template, then <end_of_turn>."""
    return solution.strip() + "<end_of_turn>"
