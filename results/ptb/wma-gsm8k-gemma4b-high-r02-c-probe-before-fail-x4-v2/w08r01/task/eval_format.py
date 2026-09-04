"""Reproduce, byte-for-byte, the strings the grader builds.

The grader (inspect_evals/gsm8k + evaluate.py) builds:
  system  = 10 few-shot blocks from the GSM8K *train* split (seed 42, shuffled)
  user    = MATH_PROMPT_TEMPLATE.format(prompt=question)
and renders them through templates/gemma3.jinja.

Everything here reads from the installed inspect_evals package so training and
grading cannot drift.
"""
from __future__ import annotations

import os

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import (
    MATH_PROMPT_TEMPLATE,
    DATASET_PATH,
    record_to_sample,
    sample_to_fewshot,
)

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "gemma3.jinja")


def fewshot_system_message(fewshot: int = 10, fewshot_seed: int = 42) -> str:
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


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.replace("{prompt}", question)


def load_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


if __name__ == "__main__":
    import hashlib
    from transformers import AutoTokenizer

    SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    tok = AutoTokenizer.from_pretrained(SNAP)
    tpl = load_template()
    print("template sha256:", hashlib.sha256(tpl.encode()).hexdigest())

    sysmsg = fewshot_system_message()
    print("system message chars:", len(sysmsg), "tokens:", len(tok(sysmsg)["input_ids"]))
    print("---- system message (first 600 chars) ----")
    print(sysmsg[:600])
    print("---- last 300 chars ----")
    print(sysmsg[-300:])

    msgs = [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": user_prompt("Natalia sold 48 clips. How many?")},
        {"role": "assistant", "content": "She sold 48.\n\nANSWER: 48"},
    ]
    rendered = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False)
    print("---- rendered tail ----")
    print(repr(rendered[-700:]))
    prompt_only = tok.apply_chat_template(msgs[:2], chat_template=tpl, tokenize=False, add_generation_prompt=True)
    print("---- prompt-only tail ----")
    print(repr(prompt_only[-500:]))
    print("prompt tokens (10-shot):", len(tok(prompt_only, add_special_tokens=False)["input_ids"]))
