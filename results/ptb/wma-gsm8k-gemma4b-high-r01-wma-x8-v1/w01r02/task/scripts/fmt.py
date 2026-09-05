"""Prompt/target formatting that mirrors the grader byte-for-byte.

Everything the grader does is reproduced here so training strings can be
compared against it:
  * MATH_PROMPT_TEMPLATE - copied verbatim from inspect_evals/gsm8k/gsm8k.py
  * render_prompt        - the same string templates/gemma3.jinja produces
  * grade                - the same extraction inspect_ai match(numeric=True) does
"""
from __future__ import annotations

import re

# verbatim from inspect_evals/gsm8k/gsm8k.py (L27-35)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

START = "<start_of_turn>"
END = "<end_of_turn>"


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(question: str, system: str | None = None) -> str:
    """Exactly what templates/gemma3.jinja emits for [system?, user] + add_generation_prompt.

    The leading <bos> is NOT included: the trainer prepends the bos token id, and
    vLLM's chat endpoint tokenizes the template output the same way.
    """
    body = user_content(question).strip()
    if system:
        body = system + "\n\n" + body
    return f"{START}user\n{body}{END}\n{START}model\n"


def render_target(solution: str, answer: str) -> str:
    return f"{solution.strip()}\n\nANSWER: {answer}{END}"


# --- grader simulation -------------------------------------------------------
# inspect_ai._util.text.strip_numeric_punctuation + scorer._common.match_str
def _strip_numeric_punctuation(s: str) -> str:
    # verbatim from inspect_ai._util.text.strip_numeric_punctuation
    stripped = re.sub(r"[$,£,€,*,_]", "", s)
    stripped = re.sub(r"\.(?=\s|$|\D)", "", stripped)
    return stripped


_LEADING_FLOAT = re.compile(r"^([+-]?\d+(?:\.\d+)?)")


def _normalize_number(number: str, precision: int = 5) -> str:
    """Mirrors inspect_ai's normalize_number + str_to_float.

    str_to_float takes the FIRST valid float in the string ("0.50.0016" -> 0.5);
    plain float() raises on it, which is what crashed the first RFT assembly pass.
    """
    if number.replace(".", "").isnumeric():
        m = _LEADING_FLOAT.match(number)
        if m is None:
            return number
        return format(float(m.group(1)), f".{precision}g")
    return number


def extract_answer(completion: str) -> str:
    """The string inspect's match(location='end', numeric=True) compares to gold."""
    v = completion.strip().casefold()
    v = _strip_numeric_punctuation(v)
    words = re.split(r"\s+", v)
    words.reverse()
    number = next((w for w in words if w.replace(".", "").isnumeric()), words[0] if words else "")
    return _normalize_number(number)


def grade(completion: str, gold: str) -> bool:
    t = _normalize_number(_strip_numeric_punctuation(gold.strip().casefold()))
    return extract_answer(completion) == t
