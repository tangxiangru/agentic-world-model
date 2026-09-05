"""Byte-exact reproduction of the grader's prompt rendering.

Everything here is derived from two files that must not be edited:
  /home/ben/task/evaluate.py                     -> which template is used
  /home/ben/task/templates/gemma3.jinja          -> how messages are rendered
  inspect_evals/gsm8k/gsm8k.py                   -> the prompt text and the few-shot block

`render_prompt` returns the exact string vLLM is fed, up to and including
"<start_of_turn>model\n".  `build_target` returns what the model must emit
after it, ending in the stop token the grader stops on.
"""
from __future__ import annotations

import hashlib
import os
import re

TASK_DIR = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "

# inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE (verbatim, already .strip()ed there)
MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

BOS = "<bos>"


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(shots) -> str:
    """shots: list of (question, reasoning, answer).  Matches sample_to_fewshot()."""
    return "\n\n".join(
        f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}" for q, r, a in shots
    )


def render_prompt(question: str, shots=None) -> str:
    """Exactly what templates/gemma3.jinja produces for
    [system(fewshot)?, user(MATH_PROMPT_TEMPLATE)] with add_generation_prompt=True."""
    prefix = ""
    if shots:
        prefix = fewshot_block(shots) + "\n\n"
    body = user_content(question).strip()
    return (
        BOS
        + "<start_of_turn>user\n"
        + prefix
        + body
        + "<end_of_turn>\n"
        + "<start_of_turn>model\n"
    )


def build_target(reasoning: str, answer: str) -> str:
    """The assistant turn: chain of thought, then a single ANSWER line, then the stop token."""
    body = reasoning.strip()
    return f"{body}\n\n{ANSWER_MARKER}{answer}{STOP_TOKEN}"


# ---- cleaning helpers -------------------------------------------------------

_CALC = re.compile(r"<<[^>]*>>")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_DOLLAR_MATH = re.compile(r"\$([^$\n]{1,80})\$")


def clean_gsm8k_reasoning(ans: str) -> tuple[str, str]:
    """openai/gsm8k train answer -> (reasoning without calculator annotations, final answer)."""
    body, _, final = ans.rpartition("####")
    return _CALC.sub("", body).strip(), final.strip().replace(",", "")


def clean_omi_solution(sol: str) -> str:
    """OpenMathInstruct-2 generated_solution -> plain prose CoT.

    The instruction in the eval prompt says a \\boxed command is not needed, and a
    second answer marker is the `double_answer_format` pitfall, so \\boxed{x} is
    unwrapped to x and inline $...$ math is left as-is.
    """
    return _BOXED.sub(r"\1", sol).strip()


def normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    return s
