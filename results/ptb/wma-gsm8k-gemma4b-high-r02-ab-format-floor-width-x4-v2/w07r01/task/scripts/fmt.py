"""Shared formatting: render prompts byte-for-byte the way the grader does.

The grader (evaluate.py) passes templates/gemma3.jinja to vLLM as the chat template,
and inspect_evals/gsm8k builds a 10-shot system message + a user turn from
MATH_PROMPT_TEMPLATE.  Anything we train on must be rendered through the *same*
jinja file (pitfall: template_unreachable).
"""
from __future__ import annotations

import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def load_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(examples) -> str:
    """examples: list of (question, reasoning, answer) -> the grader's system message."""
    return "\n\n".join(
        f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}" for q, r, a in examples
    )


# ---------------------------------------------------------------- solution cleanup
_CALC = re.compile(r"<<[^>]*>>")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_DOLLARS = re.compile(r"\$([^$\n]{1,200}?)\$")


def strip_gsm8k_calc(text: str) -> str:
    return _CALC.sub("", text)


def delatex(text: str) -> str:
    """Turn the light LaTeX OpenMathInstruct-2 uses into plain text.

    GSM8K answers are plain arithmetic; keeping $...$ and \\frac makes the model
    emit LaTeX at grading time for no benefit.
    """
    text = _BOXED.sub(r"\1", text)
    text = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1/\2)", text)
    text = re.sub(r"\\dfrac\{([^{}]*)\}\{([^{}]*)\}", r"(\1/\2)", text)
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("\\div", "/").replace("\\%", "%")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\$", "$")
    text = _DOLLARS.sub(r"\1", text)
    text = text.replace("\\[", "").replace("\\]", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_answer(a: str) -> str | None:
    """Keep only clean numeric answers - the grader is match(numeric=True)."""
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not a:
        return None
    try:
        v = float(a)
    except ValueError:
        return None
    if v == int(v) and abs(v) < 1e12:
        return str(int(v))
    return None


def build_target(solution: str, answer: str) -> str:
    """Chain of thought, then exactly one 'ANSWER: N' line, then nothing else."""
    sol = solution.strip()
    return f"{sol}\n\nANSWER: {answer}"
