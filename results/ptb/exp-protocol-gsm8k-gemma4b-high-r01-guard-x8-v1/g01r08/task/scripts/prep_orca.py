#!/usr/bin/env python3
"""Format microsoft/orca-math-word-problems-200k into the graded target format.

Orca-Math is a synthetic grade-school word-problem set (seeded from public
collections, not from the GSM8K test split). Solutions are free-form and end
with the answer stated in prose or in \\boxed{}; we keep a row only when the
last number of the solution is an unambiguous numeric answer, then re-emit it
as '... \\n\\nANSWER: <number><end_of_turn>'.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from datasets import load_dataset

sys.path.insert(0, "scripts")
from prep_data import BOXED_RE, BOXED_TAIL_RE, make_row, norm_answer  # noqa: E402

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/orca.jsonl")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    print(ds, flush=True)
    dev_q = {json.loads(l)["question"].strip() for l in open("data/dev_train300.jsonl")}

    rows, rej = [], {}

    def bump(k):
        rej[k] = rej.get(k, 0) + 1

    idx = list(range(len(ds)))
    rng.shuffle(idx)
    seen_q = set()
    for i in idx:
        if len(rows) >= args.n:
            break
        r = ds[i]
        q = r["question"].strip()
        if q in dev_q or q in seen_q:
            bump("dup")
            continue
        sol = r["answer"].strip()
        if len(sol) > args.max_chars:
            bump("long")
            continue
        prev = None
        while prev != sol:
            prev = sol
            sol = BOXED_RE.sub(r"\1", sol)
        nums = NUM.findall(sol.replace(",", ""))
        if not nums:
            bump("nonum")
            continue
        ans = norm_answer(nums[-1])
        if ans is None:
            bump("ans")
            continue
        body = BOXED_TAIL_RE.sub("", sol).strip()
        if not body:
            bump("empty")
            continue
        row = make_row(q, body, ans)
        if row is None:
            bump("make_row")
            continue
        row["src"] = "orca_math"
        seen_q.add(q)
        rows.append(row)

    print("kept", len(rows), "rejects", rej, flush=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "_for_decon.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
