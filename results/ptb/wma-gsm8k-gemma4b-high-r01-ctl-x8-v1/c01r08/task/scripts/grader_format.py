"""Single source of truth for the strings the grader actually renders.

Everything here is copied from, or derived from, the two files the grader uses:
  /usr/local/lib/python3.10/dist-packages/inspect_evals/gsm8k/gsm8k.py
  /home/ben/task/templates/gemma3.jinja

Training and probing both import this module so training and grading cannot
render different strings for the same conversation (pitfall: template_unreachable).
"""
from __future__ import annotations

import hashlib
import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# byte-for-byte the template the grader passes to vllm
with open(TEMPLATE_PATH, "rb") as _f:
    TEMPLATE_BYTES = _f.read()
TEMPLATE_SHA256 = hashlib.sha256(TEMPLATE_BYTES).hexdigest()
TEMPLATE_TEXT = TEMPLATE_BYTES.decode("utf-8")

# verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def user_text(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def _sample_to_fewshot(question: str, reasoning: str, target: str) -> str:
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def build_fewshot_system(n: int = 10, seed: int = 42) -> str:
    """Reproduce the grader's system message exactly.

    gsm8k.py does hf_dataset(openai/gsm8k, main, split=train, shuffle=True,
    seed=42, limit=10) and joins sample_to_fewshot(...) with "\\n\\n".
    inspect's hf_dataset shuffles the HF Dataset with .shuffle(seed=seed) and
    then applies the limit, i.e. the first 10 rows of the shuffled train split.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    shots = []
    for i in range(n):
        r = ds[i]
        parts = r["answer"].split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        shots.append(_sample_to_fewshot(r["question"].strip(), reasoning, target))
    return "\n\n".join(shots)


def render_prompt(question: str, system: str | None) -> str:
    """Render exactly what templates/gemma3.jinja produces for
    [system?, user] with add_generation_prompt=True, minus the bos token
    (the tokenizer adds bos itself).
    """
    content = user_text(question)
    if system:
        content = (system + "\n\n" + content).strip()
    else:
        content = content.strip()
    return f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"


def render_target(solution: str) -> str:
    """The assistant side, trimmed the way the jinja `| trim` filter would."""
    return solution.strip() + STOP_TOKEN


# ---- scorer replica: inspect_ai.scorer.match(numeric=True, location="end") ----

_NUM_PUNCT = re.compile(r"[,$%]")


def _strip_numeric_punctuation(s: str) -> str:
    # mirrors inspect_ai._util.text.strip_numeric_punctuation closely enough
    # for scoring integer gsm8k answers
    s = s.replace(",", "").replace("$", "").replace("%", "")
    return s


def _normalize_number(number: str, precision: int = 5) -> str:
    if number.replace(".", "").isnumeric():
        try:
            return format(float(number), f".{precision}g")
        except ValueError:
            return number
    return number


def score_completion(completion: str, target: str) -> bool:
    v = completion.strip().casefold()
    t = target.strip().casefold()
    if not t.isnumeric():
        return t in v
    v = _strip_numeric_punctuation(v)
    t = _normalize_number(_strip_numeric_punctuation(t))
    words = re.split(r"\s+", v)
    words.reverse()
    num = next((w for w in words if w.replace(".", "").isnumeric()), words[0] if words else "")
    return _normalize_number(num).endswith(t)


if __name__ == "__main__":
    sysmsg = build_fewshot_system()
    print("template sha256:", TEMPLATE_SHA256)
    print("fewshot system chars:", len(sysmsg))
    print("---- first 400 chars of system ----")
    print(sysmsg[:400])
    print("---- rendered zero-shot prompt ----")
    print(repr(render_prompt("What is 2+2?", None)))
