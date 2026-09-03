#!/usr/bin/env python3
"""Turn rejection-sampled completions into an SFT jsonl.

Input rows: {id, question, gold, samples:[{text, correct}]} from sample_vllm.py.
Only verified-correct chains survive. Questions the model already solves 4/4 are
downweighted: the point of the stage is the band where it is inconsistent.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import render_prompt, render_completion, load_gsm8k_train  # noqa: E402

WS = re.compile(r"\s+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--easy-keep-prob", type=float, default=0.35)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm_train = load_gsm8k_train()
    rows, hist = [], {}
    n_q = 0
    for line in open(args.samples):
        d = json.loads(line)
        n_q += 1
        k = len(d["samples"])
        good = [s["text"].strip() for s in d["samples"] if s["correct"]]
        hist[len(good)] = hist.get(len(good), 0) + 1
        if not good:
            continue
        easy = len(good) == k
        if easy and rng.random() > args.easy_keep_prob:
            continue
        seen, kept = set(), []
        rng.shuffle(good)
        for t in good:
            key = WS.sub(" ", t)
            if key in seen:
                continue
            # one answer marker, and it must be the last line
            if t.count("ANSWER:") != 1 or not re.search(r"\nANSWER: -?[\d.]+$", t):
                continue
            seen.add(key)
            kept.append(t)
            if len(kept) >= (1 if easy else args.max_per_question):
                break
        for t in kept:
            body, _, ans = t.rpartition("\nANSWER: ")
            rows.append({"q": d["question"], "r": body.strip(), "a": ans.strip()})

    rng.shuffle(rows)
    n_fs = 0
    with open(args.out, "w") as fh:
        for r in rows:
            shots = []
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(gsm_train, rng.randint(1, 4))
                n_fs += 1
            fh.write(json.dumps({
                "prompt": render_prompt(r["q"], shots),
                "completion": render_completion(r["r"], r["a"]),
                "answer": r["a"],
            }) + "\n")
    print(f"questions={n_q} rows={len(rows)} fewshot={n_fs}")
    print("correct-out-of-k histogram:", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
