"""Shared helpers: exact eval prompt format for GSM8K under inspect_evals."""
import math
import os
import re

BASE_SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

MATH_PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response '
    'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    'answer to the problem.\n\n{prompt}\n\nRemember to put your answer on its own line '
    'at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    'answer to the problem, and you do not need to use a \\boxed command.\n\nReasoning:'
)


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question.strip())


def fewshot_block(question: str, reasoning: str, target: str) -> str:
    """Mirror inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def gsm8k_fewshots(n=10, seed=42, shuffle=True):
    """Reproduce the exact few-shot system message used by the eval harness."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(n))
    blocks = []
    for r in ds:
        q = r["question"]
        parts = r["answer"].split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        blocks.append(fewshot_block(q, reasoning, target))
    return blocks


def render_prompt(tokenizer, question: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_prompt(question)})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def load_chat_template(path="templates/gemma3.jinja"):
    with open(path) as f:
        return f.read()


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    try:
        v = float(s)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(v) or abs(v) > 1e15:
        return None
    if v == int(v):
        return str(int(v))
    return str(v)


def extract_answer(text: str) -> str | None:
    """Mimic inspect's match(location='end', numeric=True): last number in output."""
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w2 = re.sub(r"[^\d\.\-]", "", w)
        n = normalize_num(w2)
        if n is not None:
            return n
    return None
