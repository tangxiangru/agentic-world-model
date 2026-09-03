#!/usr/bin/env python3
"""Problem bank for rejection sampling: {problem, answer, src} jsonl.

Two sources, both GSM8K-TRAIN-derived (never the test split):
  openmath  - the same 81k problems the SFT corpus used (gold = expected_answer)
  metamath  - meta-math/MetaMathQA GSM_Rephrased/GSM_AnsAug queries, whose
              surface form the model has never seen (gold = 'The answer is: X')
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-openmath", type=int, default=25000)
    ap.add_argument("--n-metamath", type=int, default=25000)
    ap.add_argument("--metamath-types", default="GSM_Rephrased,GSM_AnsAug")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from datasets import load_dataset

    rows = []

    if args.n_openmath:
        ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        src = ds["problem_source"]
        idx = [i for i, s in enumerate(src) if s in ("gsm8k", "augmented_gsm8k")]
        rng.shuffle(idx)
        seen = set()
        for r in ds.select(idx):
            if len(rows) >= args.n_openmath:
                break
            a = r["expected_answer"].strip().replace(",", "")
            p = r["problem"].strip()
            if not NUMERIC.match(a) or p in seen:
                continue
            seen.add(p)
            rows.append({"problem": p, "answer": a, "src": "openmath"})
        print("openmath problems:", len(rows), flush=True)

    if args.n_metamath:
        want = set(args.metamath_types.split(","))
        mm = load_dataset("meta-math/MetaMathQA", split="train")
        idx = [i for i, t in enumerate(mm["type"]) if t in want]
        rng.shuffle(idx)
        seen, n0 = set(), len(rows)
        for r in mm.select(idx):
            if len(rows) - n0 >= args.n_metamath:
                break
            resp = r["response"]
            k = resp.rfind("The answer is:")
            if k < 0:
                continue
            a = resp[k + len("The answer is:"):].strip().split()
            if not a:
                continue
            a = a[0].strip().rstrip(".").replace(",", "").replace("$", "")
            p = r["query"].strip()
            if not NUMERIC.match(a) or p in seen:
                continue
            seen.add(p)
            rows.append({"problem": p, "answer": a, "src": "metamath"})
        print("metamath problems:", len(rows) - n0, flush=True)

    rng.shuffle(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "to", args.out)


if __name__ == "__main__":
    main()
