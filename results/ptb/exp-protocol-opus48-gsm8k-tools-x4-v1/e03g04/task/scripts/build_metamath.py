#!/usr/bin/env python3
"""Build an augmented SFT set: GSM8K-train originals + a subsample of MetaMathQA
GSM-derived examples (rephrased / answer-augmented / backward). MetaMathQA is
derived from GSM8K *train* + MATH *train* (never the GSM8K test set).

All targets are collapsed to a SINGLE answer marker 'ANSWER: N<end_of_turn>'
(MetaMath ships a double marker: '#### N' and 'The answer is: N').
"""
import json
import re
import argparse
import random
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
ANS_RE = re.compile(r"The answer is:\s*(.+?)\s*$", re.DOTALL)


def clean_num(s: str) -> str | None:
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "").strip()
    # accept a plain (possibly negative / decimal) number
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        # normalize trailing .0
        if s.endswith(".0"):
            s = s[:-2]
        return s
    return None


def reformat_metamath(response: str):
    """Return (reasoning, numeric_answer) or None if answer not numeric."""
    m = ANS_RE.search(response)
    if not m:
        return None
    ans = clean_num(m.group(1))
    if ans is None:
        return None
    # reasoning = everything before "The answer is:" and before any "#### " line
    body = response[: m.start()]
    body = re.split(r"\n?####", body)[0]
    body = CALC.sub("", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body:
        return None
    return body, ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=10000)
    ap.add_argument("--gsm8k", default="data/gsm8k_train_sft.jsonl")
    ap.add_argument("--out", default="data/aug_sft.jsonl")
    ap.add_argument("--text-out", default="data/aug_text.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    ds = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_types = ["GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"]
    by_type = {t: [] for t in gsm_types}
    for i, t in enumerate(ds["type"]):
        if t in by_type:
            by_type[t].append(i)

    rows = []
    seen = set()
    kept_by_type = {}
    for t in gsm_types:
        idxs = by_type[t]
        random.shuffle(idxs)
        kept = 0
        for i in idxs:
            if kept >= args.per_type:
                break
            r = ds[i]
            out = reformat_metamath(r["response"])
            if out is None:
                continue
            reasoning, ans = out
            q = r["query"].strip()
            key = q[:120]
            if key in seen:
                continue
            seen.add(key)
            user = MATH_PROMPT_TEMPLATE.format(prompt=q)
            comp = f"{reasoning}\n\nANSWER: {ans}<end_of_turn>"
            rows.append({"prompt": user, "completion": comp, "answer": ans, "src": t})
            kept += 1
        kept_by_type[t] = kept

    # add GSM8K train originals
    n_gsm = 0
    for l in open(args.gsm8k):
        d = json.loads(l)
        rows.append({"prompt": d["prompt"], "completion": d["completion"], "answer": d["answer"], "src": "gsm8k_train"})
        n_gsm += 1

    random.shuffle(rows)
    with open(args.out, "w") as f, open(args.text_out, "w") as ft:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            ft.write(json.dumps({"text": r["prompt"] + "\n" + r["completion"]}) + "\n")

    print("kept per metamath type:", kept_by_type)
    print("gsm8k train:", n_gsm)
    print("total rows:", len(rows), "->", args.out)


if __name__ == "__main__":
    main()
