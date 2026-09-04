"""Byte-exact replication of the grader's prompt rendering and scoring.

Everything here is derived from the installed harness code so training data and
probe evaluations use the same strings the grader will use:
  * prompt template:  inspect_evals/gsm8k/gsm8k.py  MATH_PROMPT_TEMPLATE
  * few-shot prefix:  same file, sample_to_fewshot over gsm8k train,
                      shuffle(seed=42).select(range(10))  (inspect hf_dataset)
  * chat template:    /home/ben/task/templates/gemma3.jinja (what evaluate.py
                      hands to vLLM)
  * scoring:          inspect_ai.scorer._common.match_str(location="end",
                      numeric=True)
"""
from __future__ import annotations

import hashlib
import re

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"


def chat_template(path: str = TEMPLATE_PATH) -> str:
    with open(path) as f:
        return f.read()


def template_sha(path: str = TEMPLATE_PATH) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def fewshot_prefix(n: int = 10, seed: int = 42) -> str:
    """The exact system message inspect builds for gsm8k (fewshot=10, seed=42)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed).select(range(n))
    shots = []
    for r in ds:
        body, target = r["answer"].split("####")
        shots.append(
            f"{r['question']}\n\nReasoning:\n{body.strip()}\n\nANSWER: {target.strip()}"
        )
    return "\n\n".join(shots)


def render_prompt(tokenizer, question: str, prefix: str | None = None) -> str:
    """Render one graded turn exactly as vLLM will render it."""
    user = PROMPT_TEMPLATE.format(prompt=question.strip())
    messages = []
    if prefix:
        messages.append({"role": "system", "content": prefix})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template(),
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------- scoring ---
_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_last_number(text: str) -> str | None:
    """What inspect's match(location='end', numeric=True) effectively reads:
    the last whitespace-delimited token that parses as a number."""
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w = w.strip().strip("$").replace(",", "").rstrip(".").rstrip("*").strip("*")
        w = w.strip("()").strip()
        if not w:
            continue
        try:
            v = float(w)
        except ValueError:
            continue
        if v != v or v in (float("inf"), float("-inf")):  # nan/inf ("infinity", "1e999")
            continue
        return str(int(v)) if v == int(v) else str(v)
    return None


def is_correct(completion: str, gold: str) -> bool:
    got = extract_last_number(completion)
    if got is None:
        return False
    try:
        return abs(float(got) - float(str(gold).replace(",", ""))) < 1e-6
    except ValueError:
        return got == str(gold)
