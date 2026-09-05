#!/usr/bin/env python3
"""Turn sampled completions into a rejection-sampling fine-tuning set.

Input: the jsonl gen_vllm.py --mode sample writes (one line per question, a
list of completions each flagged correct/incorrect).

Keeps only completions whose final number matches the gold answer, dedupes
near-identical reasoning paths per problem (RFT's equation-set dedup, done on
the multiset of arithmetic expressions in the chain), caps the number kept per
problem, and emits rows in the same schema build_data.py writes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import END_OF_TURN, TASK_DIR, is_correct, sample_to_fewshot  # noqa: E402

EQ = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*[-+*/x×÷]\s*[-+]?\d[\d,]*(?:\.\d+)?")


def path_key(text: str) -> str:
    """Signature of a reasoning path: the sorted multiset of its arithmetic."""
    eqs = sorted(re.sub(r"\s+", "", e) for e in EQ.findall(text))
    return "|".join(eqs) if eqs else re.sub(r"\s+", " ", text.strip())[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--min-correct", type=int, default=1,
                    help="require this many of the k samples to be correct; "
                         "1-of-k admits lucky-guess chains whose reasoning is wrong")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--gsm8k-train", default=os.path.join(TASK_DIR, "data", "sft_pool_v2.jsonl"),
                    help="pool to draw the few-shot bank from (gsm8k_train rows)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, n_q, n_solved, n_comp, n_ok = [], 0, 0, 0, 0
    for path in args.samples:
        for line in open(path):
            d = json.loads(line)
            n_q += 1
            gold = str(d["answer"])
            seen: set[str] = set()
            kept = 0
            cands = []
            for c in d["completions"]:
                n_comp += 1
                t = c["text"].strip()
                if not c.get("correct") or c.get("stop") == "length":
                    continue
                n_ok += 1
                if not (args.min_chars <= len(t) <= args.max_chars):
                    continue
                if t.count("ANSWER:") != 1:
                    continue
                if not t.rstrip().split("\n")[-1].strip().startswith("ANSWER:"):
                    continue
                if not is_correct(t, gold):
                    continue
                cands.append(t)
            # a single correct sample out of k is as likely to be a chain that
            # reasons badly and lands on the answer by luck as a good one
            if len(cands) < args.min_correct:
                continue
            # prefer shorter chains among duplicates of the same path
            cands.sort(key=len)
            for t in cands:
                k = path_key(t)
                if k in seen:
                    continue
                seen.add(k)
                rows.append({"question": d["question"], "target": t + END_OF_TURN,
                             "answer": gold, "src": "rft"})
                kept += 1
                if kept >= args.max_per_problem:
                    break
            n_solved += int(kept > 0)

    # few-shot bank: the grader's sample_to_fewshot() shape, gsm8k train rows
    bank = []
    if args.fewshot_frac > 0 and os.path.exists(args.gsm8k_train):
        for line in open(args.gsm8k_train):
            r = json.loads(line)
            if r.get("src") == "gsm8k_train" and r.get("fewshots"):
                bank.extend(r["fewshots"])
            if len(bank) > 4000:
                break
    bank = list(dict.fromkeys(bank))

    rng.shuffle(rows)
    n_aug = 0
    for r in rows:
        if bank and rng.random() < args.fewshot_frac:
            r["fewshots"] = rng.sample(bank, rng.choice([2, 3, 4, 5, 8, 10]))
            n_aug += 1
        else:
            r["fewshots"] = []

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"questions {n_q}, solved at least once {n_solved} ({n_solved / max(1, n_q):.1%})")
    print(f"completions {n_comp}, correct {n_ok} ({n_ok / max(1, n_comp):.1%})")
    print(f"wrote {len(rows)} rows ({n_aug} few-shot augmented) -> {args.out}")


if __name__ == "__main__":
    main()
