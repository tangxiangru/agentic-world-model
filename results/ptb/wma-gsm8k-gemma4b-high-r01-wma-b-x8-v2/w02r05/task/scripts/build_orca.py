#!/usr/bin/env python3
"""orca-math-word-problems-200k -> prompt/completion rows.

The dataset has no answer field, only a prose solution, so the final integer is
extracted from the last line and only rows where that line contains exactly one
number are kept (measured precision on a 15-row hand check: 15/15).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import load_template, user_prompt  # noqa: E402

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
STOP = "<end_of_turn>"
NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract(ans: str):
    lines = [l for l in ans.strip().split("\n") if l.strip()]
    if not lines:
        return None
    last = lines[-1].strip()
    nums = NUM.findall(last)
    if len(nums) != 1:
        return None
    x = nums[0].replace("$", "").replace(",", "")
    try:
        f = float(x)
    except ValueError:
        return None
    if f != f or abs(f) > 1e9 or f != int(f):
        return None
    tail = last[last.rfind(nums[0]) + len(nums[0]):]
    # the number must be at the end of the sentence, modulo a unit word
    if not re.fullmatch(r"[\s\.\)%]*(?:[A-Za-z][\w\-']*\s*){0,3}[\s\.\!\)]*", tail):
        return None
    return str(int(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/orca_v1.jsonl")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    tpl = load_template()
    d = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    rng = random.Random(args.seed)

    kept = []
    for r in d:
        if len(r["answer"]) > 2200 or "ANSWER:" in r["answer"]:
            continue
        a = extract(r["answer"])
        if a is None:
            continue
        kept.append((r["question"], r["answer"].strip(), a))
    rng.shuffle(kept)
    kept = kept[: args.n]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for q, body, a in kept:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": user_prompt(q)}],
                chat_template=tpl, tokenize=False, add_generation_prompt=True,
            )
            f.write(json.dumps({
                "prompt": prompt,
                "completion": f"{body}\n\nANSWER: {a}{STOP}",
                "source": "orca_math",
                "question": q,
            }) + "\n")
    print(json.dumps({"n_rows": len(kept)}, indent=2))


if __name__ == "__main__":
    main()
