"""Prompt rendering shared by data building, training and inspection.

The whole point of this module is that training and grading must render the
*same* string. The grader (evaluate.py) passes templates/gemma3.jinja to vLLM
and inspect_evals/gsm8k builds the messages; we reproduce both here from the
same files, and hash the template so a drift is loud.
"""
from __future__ import annotations

import hashlib
import os

from jinja2 import Environment
from jinja2.sandbox import ImmutableSandboxedEnvironment

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# sha256 of templates/gemma3.jinja as shipped by the harness; if this changes the
# grader changed and every rendered prompt in data/ is stale.
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

# copied verbatim from inspect_evals/gsm8k/gsm8k.py (MATH_PROMPT_TEMPLATE)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
END_OF_TURN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_source() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_hash() -> str:
    return hashlib.sha256(template_source().encode()).hexdigest()


def _env() -> Environment:
    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg: str):  # the template calls this
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    return env


_TEMPLATE = None


def render_prompt(question: str, system: str | None = None) -> str:
    """Render exactly what vLLM will see, up to and including the model turn header."""
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _env().from_string(template_source())
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)})
    return _TEMPLATE.render(messages=messages, add_generation_prompt=True, bos_token=BOS)


def format_answer(ans: str) -> str:
    """Final-answer surface form.

    inspect's match(numeric=True) takes the last numeric whitespace token when the
    gold target is .isnumeric(); but 14 of the 1319 gsm8k test targets contain a
    thousands comma, and for those the gold is NOT .isnumeric(), so the scorer falls
    back to a literal endswith on the comma-formatted string. Emitting commas is the
    dominant strategy: commas are stripped from the completion in the numeric branch,
    and required in the literal branch.
    """
    s = ans.strip()
    try:
        neg = s.startswith("-")
        digits = s[1:] if neg else s
        if digits.isdigit() and abs(int(s)) >= 1000:
            return f"{int(s):,}"
    except ValueError:
        pass
    return s


def build_completion(solution: str, answer: str) -> str:
    return solution.rstrip() + "\n\n" + ANSWER_MARKER + format_answer(answer) + END_OF_TURN
