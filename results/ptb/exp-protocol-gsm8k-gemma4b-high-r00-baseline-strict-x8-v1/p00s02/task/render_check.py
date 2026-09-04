"""Reproduce, byte-for-byte, the prompt the grader sends to vLLM.

Pitfall `template_unreachable`: training must render the same string the grader
renders. This script builds the eval prompt with the grader's own code paths
(inspect_evals.gsm8k helpers + templates/gemma3.jinja) and prints it, plus the
token lengths that set max_seq_len.
"""
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import (
    MATH_PROMPT_TEMPLATE,
    DATASET_PATH,
    record_to_sample,
    sample_to_fewshot,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"


def fewshot_system_message(n=10, seed=42):
    fewshots = hf_dataset(
        path=DATASET_PATH,
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def main():
    tok = AutoTokenizer.from_pretrained(SNAP)
    template = open(TEMPLATE).read()
    print("template sha256:", hashlib.sha256(template.encode()).hexdigest())

    sysmsg = fewshot_system_message()
    open("/home/ben/task/analysis/fewshot_system_message.txt", "w").write(sysmsg)
    print("fewshot system message tokens:", len(tok(sysmsg)["input_ids"]))

    q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)

    msgs_eval = [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": user},
    ]
    rendered_eval = tok.apply_chat_template(
        msgs_eval, chat_template=template, tokenize=False, add_generation_prompt=True
    )
    open("/home/ben/task/analysis/rendered_eval_prompt.txt", "w").write(rendered_eval)
    print("EVAL prompt tokens (10-shot):", len(tok(rendered_eval, add_special_tokens=False)["input_ids"]))
    print("---- tail of eval prompt ----")
    print(repr(rendered_eval[-600:]))

    # zero-shot training-time rendering of the same user turn
    msgs_train = [{"role": "user", "content": user}]
    rendered_train = tok.apply_chat_template(
        msgs_train, chat_template=template, tokenize=False, add_generation_prompt=True
    )
    print("---- zero-shot training prompt ----")
    print(repr(rendered_train))
    print("zero-shot prompt tokens:", len(tok(rendered_train, add_special_tokens=False)["input_ids"]))

    # full training row (prompt + target)
    target = "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n\nANSWER: 72"
    full = tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": target}],
        chat_template=template, tokenize=False, add_generation_prompt=False,
    )
    print("---- full training row ----")
    print(repr(full))
    ids = tok(full, add_special_tokens=False)["input_ids"]
    print("full row tokens:", len(ids), "last 5 ids:", ids[-5:], "->", tok.convert_ids_to_tokens(ids[-5:]))
    print("end_of_turn id:", tok.convert_tokens_to_ids("<end_of_turn>"))


if __name__ == "__main__":
    main()
