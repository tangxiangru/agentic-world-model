#!/usr/bin/env python3
"""Shared formatting helpers: the grader's chat template and prompt template.

Every training script renders its rows through *this* file so that the string
the trainer sees is byte-for-byte the string vLLM will see at grading time
(pitfall: template_unreachable).
"""
from __future__ import annotations

import hashlib
import os

BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"

# byte-for-byte from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def read_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def load_tokenizer(path: str = BASE_SNAPSHOT):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.chat_template = read_template()
    return tok


def fewshot_system_message(n: int = 10, seed: int = 42) -> str:
    """Reproduce the grader's few-shot system message exactly.

    inspect_evals/gsm8k builds it from gsm8k main/train, shuffled with seed 42,
    limit n.  Train split only - the test split is never touched.
    """
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return "\n\n".join([sample_to_fewshot(s) for s in fewshots])


GSM8K_TRAIN_PARQUET = (
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"
)


def gsm8k_train_exemplars() -> list[str]:
    """GSM8K *train* rows rendered the way inspect_evals renders a few-shot item.

    Train split only; the test split is never read here.
    """
    import pyarrow.parquet as pq

    out = []
    for r in pq.read_table(GSM8K_TRAIN_PARQUET).to_pylist():
        parts = r["answer"].split("####")
        target = parts[-1].strip()
        reasoning = "####".join(parts[:-1]).strip()
        out.append(
            f"{r['question'].strip()}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"
        )
    return out


def render_prompt(tok, problem: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=problem)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


if __name__ == "__main__":  # quick self-check
    tok = load_tokenizer()
    print("template sha", template_sha())
    s = render_prompt(tok, "What is 2+2?")
    print(repr(s))
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print("n prompt tokens", len(ids), "first ids", ids[:5])
    comp = "2 + 2 = 4.\n\nANSWER: 4" + STOP_TOKEN
    cids = tok(comp, add_special_tokens=False)["input_ids"]
    print("n completion tokens", len(cids), "last ids", cids[-3:], tok.convert_ids_to_tokens(cids[-3:]))
    if os.environ.get("FEWSHOT"):
        sysmsg = fewshot_system_message()
        print("fewshot chars", len(sysmsg))
        print("fewshot tokens", len(tok(sysmsg, add_special_tokens=False)["input_ids"]))
        print(sysmsg[:600])
