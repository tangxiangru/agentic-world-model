"""Shared rendering helpers.

Everything here reproduces, byte for byte, what the grader builds:
  * inspect_evals/gsm8k  -> the 10-shot system message + MATH_PROMPT_TEMPLATE
  * templates/gemma3.jinja -> the chat string vLLM actually sees

Any change to the eval side must be mirrored here, so training and grading
render the same string (pitfall `template_unreachable`).
"""

from __future__ import annotations

import hashlib

# copied verbatim from inspect_evals/gsm8k/gsm8k.py (MATH_PROMPT_TEMPLATE)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

GEMMA3_TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

BOS = "<bos>"
EOT = "<end_of_turn>"


def user_content(question: str) -> str:
    """The user message the solver builds, after jinja's `| trim`."""
    return MATH_PROMPT_TEMPLATE.format(prompt=question).strip()


def fewshot_block(examples) -> str:
    """`sample_to_fewshot` joined with blank lines: the system message."""
    return "\n\n".join(
        f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}" for q, reasoning, answer in examples
    )


def render_prompt(question: str, fewshots=None) -> str:
    """gemma3.jinja with add_generation_prompt=True."""
    prefix = f"{fewshot_block(fewshots)}\n\n" if fewshots else ""
    return (
        f"{BOS}<start_of_turn>user\n"
        f"{prefix}{user_content(question)}"
        f"{EOT}\n<start_of_turn>model\n"
    )


def render_target(reasoning: str, answer: str) -> str:
    """The assistant turn, ending on the token vLLM stops at."""
    return f"{reasoning.strip()}\n\nANSWER: {answer}{EOT}"


def template_sha256(path: str = "templates/gemma3.jinja") -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
