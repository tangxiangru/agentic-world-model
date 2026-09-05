#!/usr/bin/env python3
"""Render the SFT pool into fully-templated (prompt, target) rows.

The prompt is rendered with templates/gemma3.jinja - byte-for-byte the template
evaluate.py hands to vLLM - and the user turn uses inspect_evals' own
MATH_PROMPT_TEMPLATE, so training and grading see the same string.

A `--fewshot-frac` slice of rows gets a system message of k GSM8K *train*
exemplars formatted exactly as inspect_evals.gsm8k.sample_to_fewshot does, so
the model also sees the long-prefix shape it will meet at eval time (the
harness always runs fewshot=10).
"""
from __future__ import annotations

import argparse
import json
import random

from transformers import AutoTokenizer

MODEL = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"

# copied verbatim from inspect_evals/gsm8k/gsm8k.py :: MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP = "<end_of_turn>"


def load_fewshot_bank() -> list[str]:
    """Exemplars in inspect's sample_to_fewshot format, from the GSM8K TRAIN split."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    bank = []
    for r in ds:
        reasoning, _, target = r["answer"].rpartition("####")
        bank.append(
            f"{r['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}"
        )
    return bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/home/ben/task/data/sft_pool.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human-boost", type=int, default=1,
                    help="repeat count for the human-written gsm8k train solutions")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tk = AutoTokenizer.from_pretrained(MODEL)
    template = open(TEMPLATE).read()

    pool = [json.loads(l) for l in open(args.pool)]
    human = [r for r in pool if r["source"] == "gsm8k_human"]
    rest = [r for r in pool if r["source"] != "gsm8k_human"]
    rng.shuffle(rest)

    rows = human * args.human_boost + rest[: max(0, args.n - len(human) * args.human_boost)]
    rng.shuffle(rows)

    bank = load_fewshot_bank()
    n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            user = MATH_PROMPT_TEMPLATE.format(prompt=r["problem"])
            msgs = []
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, 10)
                msgs.append({"role": "system", "content": "\n\n".join(rng.sample(bank, k))})
                n_fs += 1
            msgs.append({"role": "user", "content": user})
            prompt = tk.apply_chat_template(
                msgs, chat_template=template, tokenize=False, add_generation_prompt=True
            )
            target = r["target"].strip() + STOP
            f.write(json.dumps({"prompt": prompt, "target": target,
                                "source": r["source"]}) + "\n")
    print(f"wrote {len(rows)} rows ({n_fs} with a few-shot system prefix) -> {args.out}")


if __name__ == "__main__":
    main()
