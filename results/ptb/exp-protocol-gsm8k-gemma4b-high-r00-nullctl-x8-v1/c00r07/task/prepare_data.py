#!/usr/bin/env python3
"""Build SFT data for GSM8K from allowed sources (GSM8K *train* split + MetaMathQA).

Never touches the GSM8K test split.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

from datasets import load_dataset

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/home/ben/hf_cache/hub")

OUT_DIR = "data"
CALC = re.compile(r"<<[^>]*>>")


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    if v == int(v):
        return str(int(v))
    return str(v)


def clean_gsm_answer(ans: str) -> tuple[str, str]:
    reasoning, target = ans.split("####")
    reasoning = CALC.sub("", reasoning).strip()
    return reasoning, target.strip().replace(",", "")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(0)

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gold_by_q: dict[str, str] = {}
    orig = []
    for r in gsm:
        reasoning, target = clean_gsm_answer(r["answer"])
        n = norm_num(target)
        if n is None:
            continue
        gold_by_q[r["question"].strip()] = n
        orig.append({"question": r["question"].strip(), "solution": reasoning, "answer": n,
                     "src": "gsm8k_train"})
    print("gsm8k train:", len(orig))

    with open(f"{OUT_DIR}/gsm8k_train.jsonl", "w") as f:
        for r in orig:
            f.write(json.dumps(r) + "\n")

    # ---- MetaMathQA: GSM-derived augmentations, verified against GSM8K train gold ----
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    keep_types = {"GSM_AnsAug", "GSM_Rephrased"}
    out = []
    seen = set()
    n_nogold = n_bad = 0
    for r in mm:
        if r["type"] not in keep_types:
            continue
        resp = r["response"]
        if "The answer is:" not in resp:
            continue
        body, _, tail = resp.rpartition("The answer is:")
        pred = norm_num(tail)
        if pred is None:
            continue
        gold = gold_by_q.get(r["original_question"].strip())
        if gold is None:
            n_nogold += 1
            continue
        if gold != pred:
            n_bad += 1
            continue
        q = r["query"].strip()
        body = CALC.sub("", body).strip()
        key = (q, body)
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": q, "solution": body, "answer": pred,
                    "src": r["type"], "orig_question": r["original_question"].strip()})
    print(f"metamath gsm kept={len(out)} no_gold={n_nogold} wrong={n_bad}")
    rng.shuffle(out)
    with open(f"{OUT_DIR}/metamath_gsm.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
