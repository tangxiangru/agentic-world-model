#!/usr/bin/env python3
"""Second, disjoint slice of the same corpora (for a second pass on fresh data).

Excludes every (question, target) pair already present in the given files, so
the model sees new solutions rather than a second epoch of the same ones.
"""
from __future__ import annotations

import argparse
import json
import random

from datasets import load_dataset

from build_data import make_row

PROBE_PATH = "data/probe200.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=["data/sft_v1_eot.jsonl"])
    ap.add_argument("--n-aug", type=int, default=45000)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-human", action="store_true")
    ap.add_argument("--out", default="data/sft_v2_eot.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probe_q = {json.loads(l)["question"].strip() for l in open(PROBE_PATH)}
    seen = set()
    for p in args.exclude:
        for l in open(p):
            r = json.loads(l)
            seen.add((r["question"], r["target"].replace("<end_of_turn>", "")))
    print("excluded pairs:", len(seen))

    rows = []
    # human GSM8K train CoT again (most on-distribution; cheap, 7k rows)
    g = load_dataset("openai/gsm8k", "main")["train"]
    split = json.load(open("data/split_idx.json"))
    if not args.no_human:
        for i in sorted(split["train_idx"]):
            r = g[i]
            body, ans = r["answer"].rsplit("####", 1)
            row = make_row(r["question"], body, ans, "gsm8k_train")
            if row and row["question"] not in probe_q:
                rows.append(row)
    n_g = len(rows)

    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    omi = omi.filter(
        lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=4
    )
    pool = list(range(len(omi)))
    rng.shuffle(pool)
    kept = 0
    for i in pool:
        if kept >= args.n_aug:
            break
        r = omi[i]
        if r["problem"].strip() in probe_q or len(r["generated_solution"]) > args.max_chars:
            continue
        row = make_row(
            r["problem"], r["generated_solution"], r["expected_answer"], "omi2_" + r["problem_source"]
        )
        if row is None or (row["question"], row["target"]) in seen:
            continue
        rows.append(row)
        kept += 1
    print(f"gsm8k_train {n_g}, fresh omi2 {kept}")

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            r["target"] = r["target"].rstrip() + "<end_of_turn>"
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
