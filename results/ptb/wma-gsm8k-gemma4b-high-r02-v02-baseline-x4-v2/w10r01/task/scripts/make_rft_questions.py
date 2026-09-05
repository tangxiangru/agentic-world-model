#!/usr/bin/env python3
"""Question pool for rejection sampling: {id, question, gold} jsonl.

GSM8K train questions the SFT already saw (their gold answers are what the
filter needs) plus OpenMathInstruct-2 gsm8k questions the SFT did NOT see, so
the round-2 data is not purely a rehearsal of round 1. Held-out dev/watch items
are excluded, and the benchmark test split is never read.
"""
from __future__ import annotations

import argparse
import json
import random

from build_data import is_number, q_key
from common import TASK_DIR, norm_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TASK_DIR / "data" / "rft_questions.jsonl"))
    ap.add_argument("--seen", default=str(TASK_DIR / "data" / "sft_r1.jsonl"))
    ap.add_argument("--omi2-unseen", type=int, default=13000)
    args = ap.parse_args()

    seen = {q_key(json.loads(l)["question"]) for l in open(args.seen)}

    rows = []
    for line in (TASK_DIR / "data" / "train_pool.jsonl").open():
        r = json.loads(line)
        rows.append({"id": r["id"], "question": r["question"], "gold": norm_answer(r["gold"]), "src": "gsm8k_train"})

    from datasets import load_dataset

    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    ds = ds.filter(lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8)
    cand = []
    for i, r in enumerate(ds):
        ans = norm_answer(str(r["expected_answer"]))
        if not is_number(ans):
            continue
        k = q_key(r["problem"])
        if k in seen:
            continue
        seen.add(k)
        cand.append({"id": f"omi2u-{i}", "question": r["problem"], "gold": ans, "src": "omi2_unseen"})
    random.Random(1).shuffle(cand)
    rows.extend(cand[: args.omi2_unseen])

    random.Random(1).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter

    print(args.out, len(rows), Counter(r["src"] for r in rows))


if __name__ == "__main__":
    main()
