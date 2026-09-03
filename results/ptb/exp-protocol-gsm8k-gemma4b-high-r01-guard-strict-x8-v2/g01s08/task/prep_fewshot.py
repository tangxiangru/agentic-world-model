#!/usr/bin/env python3
"""Re-render existing SFT rows with a random GSM8K-TRAIN fewshot block in front.

The grader prompts with a system message holding 10 solved GSM8K TRAIN examples,
which the gemma template folds into the first user turn. exp-03 was trained with
one problem per prompt and therefore continues the in-context pattern instead of
ending its turn. These rows put k ~ U{0..10} solved examples in front of the same
targets, rendered by the grader's own template.
"""
from __future__ import annotations

import argparse
import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"


def fewshot_pool():
    """Mirror inspect_evals.gsm8k.record_to_sample + sample_to_fewshot on TRAIN."""
    out = []
    for r in load_dataset("openai/gsm8k", "main")["train"]:
        parts = r["answer"].split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        out.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/sft_omi2_x2.jsonl")
    ap.add_argument("--out", default="data/sft_fewshot.jsonl")
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N source rows")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = open(TEMPLATE).read()
    pool = fewshot_pool()
    print(f"fewshot pool: {len(pool)} GSM8K train examples")

    rows, kept, dropped = [], 0, 0
    ks = []
    with open(args.src) as fh:
        for i, line in enumerate(fh):
            if i < args.skip:
                continue
            if kept >= args.n:
                break
            r = json.loads(line)
            # the harness always uses 10, so weight k=10; keep k=0 so the
            # zero-shot behaviour survives, and spread the rest for robustness
            u = rng.random()
            k = 0 if u < 0.2 else (10 if u < 0.5 else rng.randint(1, 9))
            user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
            msgs = []
            if k:
                msgs.append({"role": "system", "content": "\n\n".join(rng.sample(pool, k))})
            msgs.append({"role": "user", "content": user})
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            n_tok = len(tok(prompt + r["completion"], add_special_tokens=False)["input_ids"])
            if n_tok > args.max_seq_len:
                dropped += 1
                continue
            ks.append(k)
            kept += 1
            rows.append({"question": r["question"], "answer": r["answer"],
                         "prompt": prompt, "completion": r["completion"], "k_fewshot": k})

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({dropped} dropped over {args.max_seq_len} tokens)"
          f"; mean k={sum(ks)/len(ks):.2f}")


if __name__ == "__main__":
    main()
