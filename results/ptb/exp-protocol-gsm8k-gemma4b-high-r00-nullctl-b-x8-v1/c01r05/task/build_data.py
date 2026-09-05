#!/usr/bin/env python3
"""Build SFT training data for GSM8K, formatted to match the inspect_evals prompt."""
from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset

from common import user_prompt, fewshot_system_message

CALC_RE = re.compile(r"<<[^>]*>>")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_num(s: str) -> str:
    s = s.replace(",", "").strip()
    if s.endswith("."):
        s = s[:-1]
    try:
        f = float(s)
    except ValueError:
        return s
    if f == int(f):
        return str(int(f))
    return ("%f" % f).rstrip("0").rstrip(".")


def clean_reasoning(text: str) -> str:
    text = CALC_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_record(question: str, reasoning: str, answer: str, source: str):
    answer = norm_num(answer)
    if not answer:
        return None
    reasoning = clean_reasoning(reasoning)
    if not reasoning:
        return None
    return {
        "question": question.strip(),
        "response": f"{reasoning}\n\nANSWER: {answer}",
        "answer": answer,
        "source": source,
    }


def gsm8k_train_records():
    ds = load_dataset("openai/gsm8k", "main")["train"]
    out = []
    for r in ds:
        reasoning, _, ans = r["answer"].partition("####")
        rec = make_record(r["question"], reasoning, ans, "gsm8k_train")
        if rec:
            out.append(rec)
    return out


def gsm8k_socratic_records():
    ds = load_dataset("openai/gsm8k", "socratic")["train"]
    out = []
    for r in ds:
        reasoning, _, ans = r["answer"].partition("####")
        rec = make_record(r["question"], reasoning, ans, "gsm8k_socratic")
        if rec:
            out.append(rec)
    return out


ANS_TAIL_RE = re.compile(r"The answer is:?\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def metamath_records(max_per_type: dict[str, int], seed: int = 0):
    ds = load_dataset("meta-math/MetaMathQA")["train"]
    rng = random.Random(seed)
    buckets: dict[str, list] = {k: [] for k in max_per_type}
    for r in ds:
        t = r["type"]
        if t not in buckets:
            continue
        buckets[t].append(r)
    out = []
    for t, rows in buckets.items():
        rng.shuffle(rows)
        n = 0
        for r in rows:
            if n >= max_per_type[t]:
                break
            resp = r["response"]
            m = ANS_TAIL_RE.search(resp)
            if not m:
                continue
            ans = m.group(1).strip()
            if not NUM_RE.fullmatch(ans.replace("$", "").strip()):
                continue
            ans = norm_num(ans.replace("$", ""))
            body = resp[: m.start()]
            body = re.sub(r"####.*$", "", body, flags=re.DOTALL).strip()
            if not body:
                continue
            rec = make_record(r["query"], body, ans, t)
            if rec:
                out.append(rec)
                n += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--metamath-ansaug", type=int, default=30000)
    ap.add_argument("--metamath-rephrased", type=int, default=30000)
    ap.add_argument("--metamath-sv", type=int, default=8000)
    ap.add_argument("--metamath-fobar", type=int, default=8000)
    ap.add_argument("--gsm-repeat", type=int, default=2)
    ap.add_argument("--socratic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = []
    base = gsm8k_train_records()
    for _ in range(args.gsm_repeat):
        recs.extend(base)
    if args.socratic:
        recs.extend(gsm8k_socratic_records())
    recs.extend(
        metamath_records(
            {
                "GSM_AnsAug": args.metamath_ansaug,
                "GSM_Rephrased": args.metamath_rephrased,
                "GSM_SV": args.metamath_sv,
                "GSM_FOBAR": args.metamath_fobar,
            },
            seed=args.seed,
        )
    )

    rng = random.Random(args.seed)
    rng.shuffle(recs)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    from collections import Counter

    print(Counter(r["source"] for r in recs))
    print("total", len(recs), "->", args.out)


if __name__ == "__main__":
    main()
