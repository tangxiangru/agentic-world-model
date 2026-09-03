#!/usr/bin/env python3
"""Question pools for rejection sampling: {question, gold} jsonl.

Pool A: the GSM8K train split (human gold).
Pool B: OpenMathInstruct-2 gsm8k/augmented_gsm8k problems from shards not used
        to build data/train_sft.jsonl, so RFT sees problems the SFT run did not.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pyarrow.parquet as pq

OMI_DIR = ("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
           "snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data")
GSM8K_TRAIN = ("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
               "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet")
NUM_RE = re.compile(r"^\d[\d,]*(\.\d+)?$")


def clean(s: str) -> str | None:
    s = s.strip().replace(",", "").rstrip(".")
    if not NUM_RE.match(s) or len(s) > 12:
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", default="data/train_sft.jsonl")
    ap.add_argument("--shards", default="4,5,6,7,8,9")
    ap.add_argument("--max-new", type=int, default=40000)
    ap.add_argument("--out-a", default="data/pool_gsm8k_train.jsonl")
    ap.add_argument("--out-b", default="data/pool_fresh.jsonl")
    args = ap.parse_args()

    seen = set()
    if os.path.exists(args.seen):
        for l in open(args.seen):
            seen.add(json.loads(l)["question"].strip())
    print("seen questions:", len(seen))

    with open(args.out_a, "w") as f:
        n = 0
        for r in pq.read_table(GSM8K_TRAIN).to_pylist():
            g = clean(r["answer"].split("####")[-1])
            if g is None:
                continue
            f.write(json.dumps({"question": r["question"].strip(), "gold": g,
                                "src": "gsm8k_train"}) + "\n")
            n += 1
    print("pool A:", n)

    out, seen_new = [], set()
    for i in [int(x) for x in args.shards.split(",")]:
        path = f"{OMI_DIR}/train-{i:05d}-of-00032.parquet"
        if not os.path.exists(path):
            print("missing shard", path)
            continue
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=20000,
                                 columns=["problem", "expected_answer", "problem_source"]):
            src = b.column("problem_source").to_pylist()
            prob = b.column("problem").to_pylist()
            ans = b.column("expected_answer").to_pylist()
            for s, p, a in zip(src, prob, ans):
                if s not in ("gsm8k", "augmented_gsm8k"):
                    continue
                p = p.strip()
                if p in seen or p in seen_new:
                    continue
                g = clean(a)
                if g is None:
                    continue
                seen_new.add(p)
                out.append({"question": p, "gold": g, "src": s})
                if len(out) >= args.max_new:
                    break
            if len(out) >= args.max_new:
                break
        print(f"shard {i}: pool B {len(out)}")
        if len(out) >= args.max_new:
            break

    with open(args.out_b, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("pool B:", len(out))


if __name__ == "__main__":
    main()
