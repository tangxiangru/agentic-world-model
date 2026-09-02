"""Shared formatting helpers.

Everything in here mirrors, byte for byte, what the grader does:
  * MATH_PROMPT_TEMPLATE / sample_to_fewshot come from
    inspect_evals/gsm8k/gsm8k.py
  * the chat rendering comes from templates/gemma3.jinja
Both are hash-checked by scripts/check_template.py.
"""
from __future__ import annotations

import re

# --- verbatim from inspect_evals/gsm8k/gsm8k.py -----------------------------
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"


def user_prompt(question: str, fewshot_block: str | None = None) -> str:
    """The user-message content exactly as gemma3.jinja will see it.

    gemma3.jinja folds a system message into the first user turn as
    `system_content + "\\n\\n"` and then appends `content | trim`.
    """
    body = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    if fewshot_block:
        return fewshot_block + "\n\n" + body
    return body


def fewshot_example(question: str, reasoning: str, answer: str) -> str:
    """Verbatim from gsm8k.py::sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def render_prompt(user_text: str) -> str:
    """gemma3.jinja with add_generation_prompt=True, one user turn."""
    return f"{BOS}{SOT}user\n{user_text.strip()}{EOT}\n{SOT}model\n"


def render_completion(answer_text: str) -> str:
    """The model turn plus the terminator the grader stops on."""
    return f"{answer_text.strip()}{EOT}\n"


# --- solution cleanup -------------------------------------------------------
_BOXED = "\\boxed"


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (handles nesting)."""
    out = []
    i = 0
    while True:
        j = text.find(_BOXED, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = j + len(_BOXED)
        while k < len(text) and text[k] == " ":
            k += 1
        if k >= len(text) or text[k] != "{":
            out.append(_BOXED)
            i = j + len(_BOXED)
            continue
        depth = 0
        k0 = k
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append(text[k0 + 1 : k])
        i = k + 1


NUM_RE = re.compile(r"^-?\d{1,12}(?:,\d{3})*(?:\.\d+)?$")


def clean_number(s: str) -> str | None:
    s = s.strip().replace("$", "").replace(",", "").rstrip(".")
    if not NUM_RE.match(s):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


CALC_RE = re.compile(r"<<[^>]*>>")


def strip_calc(text: str) -> str:
    """Remove the gsm8k `<<3*2=6>>` calculator annotations."""
    return CALC_RE.sub("", text)
