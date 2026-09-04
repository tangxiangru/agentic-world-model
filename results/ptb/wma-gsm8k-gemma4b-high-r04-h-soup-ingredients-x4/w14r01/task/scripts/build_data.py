#!/usr/bin/env python3
"""Build the SFT corpus.

Sources (all GSM8K *train* derived or MATH train derived - never the test split):
  * openai/gsm8k  main/train          - 7,473 human-written CoTs
  * meta-math/MetaMathQA  GSM_*       - augmentations of the same train questions

Every target is reshaped into exactly one answer marker ("ANSWER: N") so the
grader's location='end' numeric matcher reads the right number, and the
"#### N" / "The answer is: N" markers that MetaMathQA carries are removed
(pitfall double_answer_format).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ANSWER_MARKER,
    STOP_TOKEN,
    normalize_number,
    sample_to_fewshot,
    strip_calc_annotations,
    user_text,
)

from datasets import load_dataset  # noqa: E402


def clean_gsm8k_train(ds):
    rows = []
    for r in ds:
        ans = r["answer"]
        if "####" not in ans:
            continue
        body, final = ans.rsplit("####", 1)
        n = normalize_number(final)
        if n is None:
            continue
        body = strip_calc_annotations(body).strip()
        body = re.sub(r"[ \t]+", " ", body)
        rows.append(
            {
                "question": r["question"].strip(),
                "reasoning": body,
                "answer": n,
                "src": "gsm8k_train",
            }
        )
    return rows


_ANS_IS = re.compile(r"\nThe answer is:.*\Z", re.S)
_HASH = re.compile(r"\n?####[^\n]*")


def clean_metamath(ds, keep_types, per_type_cap, rng):
    buckets: dict[str, list] = {t: [] for t in keep_types}
    for r in ds:
        t = r["type"]
        if t not in buckets:
            continue
        resp = r["response"]
        m = re.search(r"The answer is:\s*(.+?)\s*\Z", resp, re.S)
        if not m:
            continue
        n = normalize_number(m.group(1))
        if n is None:
            continue
        body = _ANS_IS.sub("", resp)
        body = _HASH.sub("", body).strip()
        if not body or "\\boxed" in body or "####" in body:
            continue
        body = strip_calc_annotations(body).strip()
        buckets[t].append(
            {
                "question": r["query"].strip(),
                "reasoning": body,
                "answer": n,
                "src": t,
            }
        )
    out = []
    for t, rows in buckets.items():
        rng.shuffle(rows)
        cap = per_type_cap.get(t, 0)
        out.extend(rows[:cap])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gsm-repeat", type=int, default=2)
    ap.add_argument("--ansaug", type=int, default=26000)
    ap.add_argument("--rephrased", type=int, default=16000)
    ap.add_argument("--sv", type=int, default=4000)
    ap.add_argument("--fobar", type=int, default=4000)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--max-fewshot-k", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm = load_dataset("openai/gsm8k", "main")["train"]
    gsm_rows = clean_gsm8k_train(gsm)
    print(f"gsm8k_train usable: {len(gsm_rows)}", file=sys.stderr)

    mm = load_dataset("meta-math/MetaMathQA")["train"]
    caps = {
        "GSM_AnsAug": args.ansaug,
        "GSM_Rephrased": args.rephrased,
        "GSM_SV": args.sv,
        "GSM_FOBAR": args.fobar,
    }
    mm_rows = clean_metamath(mm, set(caps), caps, rng)
    print(f"metamath usable: {len(mm_rows)}", file=sys.stderr)

    pool = gsm_rows * args.gsm_repeat + mm_rows

    # exact-duplicate removal on (question, reasoning)
    seen = set()
    uniq = []
    dup = 0
    for r in pool:
        k = (r["question"], r["reasoning"])
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        uniq.append(r)
    print(f"pool {len(pool)} -> unique {len(uniq)} (dropped {dup})", file=sys.stderr)

    # few-shot exemplar pool: original human GSM8K train CoTs only
    exemplars = gsm_rows

    rng.shuffle(uniq)
    n_fs = int(len(uniq) * args.fewshot_frac)
    out_rows = []
    with open(args.out, "w") as f:
        for i, r in enumerate(uniq):
            block = None
            if i < n_fs:
                k = rng.choice([2, 4, args.max_fewshot_k])
                picks = rng.sample(exemplars, k)
                block = "\n\n".join(
                    sample_to_fewshot(p["question"], p["reasoning"], p["answer"])
                    for p in picks
                )
            completion = (
                r["reasoning"].strip() + "\n\n" + ANSWER_MARKER + r["answer"] + STOP_TOKEN
            )
            out_rows.append(
                {
                    "prompt": user_text(r["question"], block),
                    "completion": completion,
                    "question": r["question"],
                    "answer": r["answer"],
                    "src": r["src"],
                    "fewshot": block is not None,
                }
            )
        # interleave the few-shot rows with the rest so any prefix of the file
        # is a representative sample (and so length stats are not biased)
        rng.shuffle(out_rows)
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(uniq)} rows to {args.out} ({n_fs} few-shot)", file=sys.stderr)


if __name__ == "__main__":
    main()
