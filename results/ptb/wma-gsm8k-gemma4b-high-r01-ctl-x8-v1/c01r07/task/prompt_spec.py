"""Single source of truth for the exact strings the grader renders.

Everything here is read out of the installed harness (inspect_evals/gsm8k) and
templates/gemma3.jinja, so training data cannot drift from grading.
"""
from __future__ import annotations

import hashlib
import os
import re

from inspect_evals.gsm8k.gsm8k import (
    MATH_PROMPT_TEMPLATE,
    record_to_sample,
    sample_to_fewshot,
)
from inspect_ai.dataset import hf_dataset

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "gemma3.jinja")
DATASET_PATH = "openai/gsm8k"

BOS = "<bos>"
START = "<start_of_turn>"
END = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fewshot_system_message(fewshot: int = 10, fewshot_seed: int = 42) -> str:
    """Byte-identical to what inspect_evals.gsm8k builds as the system message."""
    fewshots = hf_dataset(
        path=DATASET_PATH,
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=fewshot_seed,
        limit=fewshot,
    )
    return "\n\n".join([sample_to_fewshot(s) for s in fewshots])


def user_content(question: str) -> str:
    """What prompt_template(MATH_PROMPT_TEMPLATE) produces for one item."""
    return MATH_PROMPT_TEMPLATE.replace("{prompt}", question)


def render_prompt(question: str, system: str | None) -> str:
    """Reproduce templates/gemma3.jinja for [system?, user] + add_generation_prompt."""
    prefix = (system + "\n\n") if system else ""
    return (
        BOS
        + START + "user\n"
        + prefix
        + user_content(question).strip()
        + END + "\n"
        + START + "model\n"
    )


def render_target(reasoning: str, answer: str) -> str:
    """The assistant turn the grader wants, terminated the way the template does."""
    return reasoning.strip() + "\n\n" + ANSWER_MARKER + answer.strip() + END + "\n"


# ---- solution-body cleaning -------------------------------------------------

_CALC = re.compile(r"<<[^>]*>>")
_HASH = re.compile(r"\n?####.*$", re.S)
_BOXED_TAIL = re.compile(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?\\?boxed\{[^}]*\}\.?\s*$", re.I)
_ANS_IS_TAIL = re.compile(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*[^\n]*$", re.I)


def strip_boxed(text: str) -> str:
    """Turn \\boxed{x} into x everywhere (balanced-brace aware)."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        out.append(text[j + len("\\boxed{"): k - 1])
        i = k


def clean_body(text: str) -> str:
    """Remove every competing answer marker so exactly one 'ANSWER:' survives."""
    t = _CALC.sub("", text)
    t = _HASH.sub("", t)          # gsm8k's own '#### N'
    t = strip_boxed(t)
    t = t.replace("$\\boxed", "").replace("\\boxed", "")
    t = _BOXED_TAIL.sub("", t)
    t = _ANS_IS_TAIL.sub("", t.rstrip())   # MetaMath's 'The answer is: N'
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"^ANSWER:.*$", "", t, flags=re.M)
    return t.strip()


if __name__ == "__main__":
    sysmsg = fewshot_system_message()
    print("template sha256:", template_sha256())
    print("system message chars:", len(sysmsg))
    print("---- system head ----")
    print(sysmsg[:600])
    print("---- system tail ----")
    print(sysmsg[-400:])
    print("---- rendered prompt (zero-shot) ----")
    print(repr(render_prompt("What is 2+2?", None)))
