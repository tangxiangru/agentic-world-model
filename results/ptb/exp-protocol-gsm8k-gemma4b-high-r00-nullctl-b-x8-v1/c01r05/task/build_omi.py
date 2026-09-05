#!/usr/bin/env python3
"""Extract the GSM8K-flavoured slice of OpenMathInstruct-2 into our SFT format."""
from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_from_disk

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NUM_RE = re.compile(r"^-?\d[\d,]*\.?\d*$")


def norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    if s.endswith("."):
        s = s[:-1]
    try:
        f = float(s)
    except ValueError:
        return ""
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return ("%.6f" % f).rstrip("0").rstrip(".")


def clean_solution(sol: str) -> str:
    sol = BOXED_RE.sub(r"\1", sol)
    sol = sol.replace("\\boxed", "")
    sol = re.sub(r"[ \t]+", " ", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/omi_1m")
    ap.add_argument("--out", default="data/omi_gsm.jsonl")
    ap.add_argument("--max-aug", type=int, default=110000)
    ap.add_argument("--max-orig", type=int, default=15000)
    ap.add_argument("--max-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_from_disk(args.path)
    rng = random.Random(args.seed)
    aug, orig = [], []
    for r in d:
        src = r["problem_source"]
        if src not in ("gsm8k", "augmented_gsm8k"):
            continue
        ans = r["expected_answer"].strip()
        if not NUM_RE.match(ans.replace("$", "")):
            continue
        ans = norm_num(ans)
        if not ans:
            continue
        sol = r["generated_solution"]
        if "\\boxed" not in sol:
            continue
        sol = clean_solution(sol)
        if len(sol) > args.max_chars or len(sol) < 30:
            continue
        rec = {
            "question": r["problem"].strip(),
            "response": f"{sol}\n\nANSWER: {ans}",
            "answer": ans,
            "source": "omi_" + src,
        }
        (aug if src == "augmented_gsm8k" else orig).append(rec)

    rng.shuffle(aug)
    rng.shuffle(orig)
    recs = aug[: args.max_aug] + orig[: args.max_orig]
    rng.shuffle(recs)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    from collections import Counter

    print(Counter(r["source"] for r in recs), "total", len(recs), "->", args.out)


if __name__ == "__main__":
    main()
