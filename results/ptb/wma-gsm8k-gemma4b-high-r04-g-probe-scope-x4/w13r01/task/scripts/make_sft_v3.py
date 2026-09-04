#!/usr/bin/env python3
"""data/sft_v3_raw.jsonl -> data/sft_v3.jsonl, the corpus exp-03 trains on.

Same two post-processing steps as make_sft_v2.py (normalise the terminator, then
subsample to a quota with the long few-shot rows capped), but with exp-03's
larger quota. Kept as a separate file so data/sft_v2.jsonl stays reproducible
from the script exp-02's card names.

Deterministic: seeded rng, no network, no clock.
"""
from __future__ import annotations

import argparse
import json
import random

NOSYS = "<bos><start_of_turn>user\nSolve the following math problem"
QUOTA = {
    "gsm8k_train_gold": 7473,
    "gsm8k": 14757,
    "augmented_gsm8k": 85000,
    "augmented_math": 11000,
    "math": 3320,
}
FEWSHOT_QUOTA = 4800


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/sft_v3_raw.jsonl")
    ap.add_argument("--out", default="data/sft_v3.jsonl")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rows, dropped = [], 0
    with open(args.src) as f:
        for line in f:
            o = json.loads(line)
            if o["completion"].endswith("<end_of_turn>\n"):
                o["completion"] = o["completion"][:-1]
                o["text"] = o["prompt"] + o["completion"]
                o["n_tokens"] -= 1
            if o["completion"].count("ANSWER:") != 1:
                dropped += 1
                continue
            assert o["completion"].endswith("<end_of_turn>")
            o["fewshot"] = not o["prompt"].startswith(NOSYS)
            rows.append(o)
    print(f"normalised {len(rows)} rows; dropped {dropped} without exactly one marker")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    sel, got, fs = [], {k: 0 for k in QUOTA}, 0
    for r in rows:
        s = r["source"]
        if got.get(s, 0) >= QUOTA.get(s, 0):
            continue
        if r["fewshot"]:
            if fs >= FEWSHOT_QUOTA:
                continue
            fs += 1
        got[s] += 1
        sel.append(r)
    rng.shuffle(sel)

    print(f"selected {len(sel)} {got} fewshot={fs}")
    print(f"tokens {sum(r['n_tokens'] for r in sel)/1e6:.2f}M  "
          f"max {max(r['n_tokens'] for r in sel)}")

    with open(args.out, "w") as f:
        for r in sel:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "_check.jsonl"), "w") as f:
        for r in sel:
            f.write(json.dumps({"text": r["question"] + "\n" + r["completion"]}) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
