#!/usr/bin/env python3
"""Render the exact prompt the harness sends, so training can match it byte-for-byte.

Writes:
  data/fewshot_system.txt   the 10-shot system message inspect_evals.gsm8k builds
  data/rendered_example.txt one full prompt as vLLM receives it
"""
import os

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import (
    MATH_PROMPT_TEMPLATE,
    record_to_sample,
    sample_to_fewshot,
)
from transformers import AutoTokenizer

SNAP = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fewshot_system() -> str:
    fs = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in fs)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(SNAP)
    tmpl = open(os.path.join(ROOT, "templates/gemma3.jinja")).read()
    sysmsg = fewshot_system()
    with open(os.path.join(ROOT, "data/fewshot_system.txt"), "w") as f:
        f.write(sysmsg)
    q = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether in April and May?"
    )
    msgs = [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": MATH_PROMPT_TEMPLATE.replace("{prompt}", q)},
    ]
    rendered = tok.apply_chat_template(
        msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True
    )
    with open(os.path.join(ROOT, "data/rendered_example.txt"), "w") as f:
        f.write(rendered)
    print("system tokens", len(tok(sysmsg)["input_ids"]))
    print("prompt tokens", len(tok(rendered, add_special_tokens=False)["input_ids"]))


if __name__ == "__main__":
    main()
