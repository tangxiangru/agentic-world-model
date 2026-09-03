#!/usr/bin/env python3
"""Re-render a fraction of an SFT jsonl with a GSM8K-TRAIN fewshot prefix.

Same rendering as prep_fewshot.py, but the fewshot share is a flag, so a
rejection-sampling file (built with zero-shot prompts) can be mixed to keep the
graded 10-shot condition represented without paying prefix tokens on every row.
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
    out = []
    for r in load_dataset("openai/gsm8k", "main")["train"]:
        parts = r["answer"].split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        out.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--p-fewshot", type=float, default=0.35)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = open(TEMPLATE).read()
    pool = fewshot_pool()

    rows, dropped, n_fs = [], 0, 0
    with open(args.src) as fh:
        for line in fh:
            if len(rows) >= args.n:
                break
            r = json.loads(line)
            if rng.random() < args.p_fewshot:
                # 2/3 of the prefixed rows use k=10, the graded condition
                k = 10 if rng.random() < 0.66 else rng.randint(1, 9)
                msgs = [{"role": "system", "content": "\n\n".join(rng.sample(pool, k))},
                        {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                n_fs += 1
            else:
                prompt = r["prompt"]
                k = 0
            if len(tok(prompt + r["completion"], add_special_tokens=False)["input_ids"]) > args.max_seq_len:
                dropped += 1
                continue
            out = dict(r)
            out["prompt"] = prompt
            out["k_fewshot"] = k
            rows.append(out)

    rng.shuffle(rows)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows ({n_fs} with a fewshot prefix, {dropped} dropped) -> {args.out}")


if __name__ == "__main__":
    main()
