#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Sources (all derived from the GSM8K TRAIN split or from public synthetic
augmentations of it; the GSM8K test split is never read here):
  * openai/gsm8k train, minus the 200 held-out probe items  (human CoT)
  * nvidia/OpenMathInstruct-2 train_1M, problem_source in {gsm8k, augmented_gsm8k}

Every target is rewritten to the exact shape the grader reads:
    <reasoning>

    ANSWER: <number>
so that the last whitespace-separated numeric token of the completion is the
graded answer (inspect_ai match(location="end", numeric=True)).
"""
from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset

PROBE_PATH = "data/probe200.jsonl"
ANSWER_MARKER = "ANSWER: "

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
FINAL_ANS_SENT_RE = re.compile(
    r"(?:so |thus |therefore |hence )?(?:the )?(?:final )?answer is[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
CALC_RE = re.compile(r"<<[^>]*>>")


def is_plain_number(s: str) -> bool:
    s = s.strip().replace(",", "").replace("$", "")
    if s.startswith("-"):
        s = s[1:]
    return bool(s) and s.replace(".", "", 1).isdigit()


def norm_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def clean_body(body: str) -> str:
    """Strip trailing 'the answer is ...' / boxed noise so the marker is unique."""
    body = CALC_RE.sub("", body)
    body = BOXED_RE.sub(r"\1", body)
    body = body.replace("ANSWER:", "answer:")
    # drop a trailing "the final answer is X" sentence/line
    lines = [ln.rstrip() for ln in body.strip().split("\n")]
    while lines and (
        not lines[-1].strip()
        or FINAL_ANS_SENT_RE.search(lines[-1].strip())
        or lines[-1].strip().startswith("####")
    ):
        lines.pop()
    body = "\n".join(lines).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def make_row(question: str, body: str, answer: str, src: str):
    body = clean_body(body)
    ans = norm_number(answer)
    if not body or not is_plain_number(ans):
        return None
    if len(body) < 20:
        return None
    target = f"{body}\n\n{ANSWER_MARKER}{ans}"
    if target.count(ANSWER_MARKER) != 1:
        return None
    return {"question": question.strip(), "target": target, "source": src, "answer": ans}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-aug", type=int, default=45000)
    ap.add_argument("--n-omi-gsm8k", type=int, default=15000)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/sft_v1.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probe_q = {json.loads(l)["question"].strip() for l in open(PROBE_PATH)}
    split = json.load(open("data/split_idx.json"))
    train_idx = set(split["train_idx"])

    rows = []

    # --- 1. human-written GSM8K train CoT ---------------------------------
    g = load_dataset("openai/gsm8k", "main")["train"]
    n_gsm = 0
    for i in sorted(train_idx):
        r = g[i]
        body, ans = r["answer"].rsplit("####", 1)
        row = make_row(r["question"], body, ans, "gsm8k_train")
        if row and row["question"] not in probe_q:
            for _ in range(args.gsm8k_repeat):
                rows.append(row)
            n_gsm += 1
    print(f"gsm8k_train: {n_gsm} problems x{args.gsm8k_repeat}")

    # --- 2. OpenMathInstruct-2 (gsm8k-derived only) ------------------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    omi = omi.filter(
        lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"),
        num_proc=8,
    )
    idx_by_src = {"gsm8k": [], "augmented_gsm8k": []}
    for i, s in enumerate(omi["problem_source"]):
        idx_by_src[s].append(i)
    for src, want in (("gsm8k", args.n_omi_gsm8k), ("augmented_gsm8k", args.n_aug)):
        pool = idx_by_src[src]
        rng.shuffle(pool)
        kept = 0
        for i in pool:
            if kept >= want:
                break
            r = omi[i]
            if r["problem"].strip() in probe_q:
                continue
            if len(r["generated_solution"]) > args.max_chars:
                continue
            row = make_row(
                r["problem"], r["generated_solution"], r["expected_answer"], "omi2_" + src
            )
            if row:
                rows.append(row)
                kept += 1
        print(f"omi2_{src}: {kept}")

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
