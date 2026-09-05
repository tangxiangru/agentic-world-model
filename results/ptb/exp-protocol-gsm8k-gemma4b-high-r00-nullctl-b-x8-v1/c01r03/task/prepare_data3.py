#!/usr/bin/env python3
"""Stage-2 SFT data: fresh targets and fresh phrasings for the same domain.

  * OpenMathInstruct-2 solutions that were *not* used in stage 1 (new reasoning
    traces for the same augmented-GSM8K problem pool).
  * MetaMathQA GSM subsets (rephrasings / backward variants of GSM8K *train*).
  * A second pass over the human GSM8K train solutions.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_dataset

from prepare_data import MATH_PROMPT_TEMPLATE, clean_answer, fewshot_block, gsm8k_train_rows, strip_boxed

FULL_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-000*.parquet"
INT_RE = re.compile(r"^-?\d+$")


def omi2_fresh(used_pairs, n_target, max_extra_per_problem=2):
    out = []
    per = defaultdict(int)
    for f in sorted(glob.glob(FULL_GLOB)):
        t = pq.read_table(f)
        t = t.filter(
            pc.is_in(t.column("problem_source"), value_set=pa.array(["gsm8k", "augmented_gsm8k"]))
        )
        for r in t.to_pylist():
            q = r["problem"].strip()
            if per[q] >= max_extra_per_problem:
                continue
            a = clean_answer(r["expected_answer"])
            if a is None:
                continue
            sol = r["generated_solution"]
            if len(sol) > 2500 or len(q) > 1200:
                continue
            sol = strip_boxed(sol)
            if sol is None:
                continue
            sol = sol.strip()
            if "\\boxed" in sol or len(sol) < 40:
                continue
            if (q, sol) in used_pairs:
                continue
            per[q] += 1
            out.append({"question": q, "solution": sol, "answer": a, "source": "omi2_fresh"})
        print(f"  omi2 fresh: {len(out)}", flush=True)
        if len(out) >= n_target:
            break
    random.Random(0).shuffle(out)
    return out[:n_target]


def metamath_gsm(quota: dict):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    got = defaultdict(int)
    out = []
    for r in ds:
        ty = r["type"]
        if ty not in quota or got[ty] >= quota[ty]:
            continue
        resp = r["response"]
        if "The answer is:" not in resp:
            continue
        body, ans = resp.rsplit("The answer is:", 1)
        ans = ans.strip().replace(",", "").replace("$", "")
        if not INT_RE.match(ans):
            continue
        body = re.sub(r"####\s*[-\d,\.]+\s*$", "", body.strip()).strip()
        body = strip_boxed(body) if "\\boxed" in body else body
        if body is None or len(body) < 30:
            continue
        got[ty] += 1
        out.append(
            {
                "question": r["query"].strip(),
                "solution": body.strip(),
                "answer": str(int(ans)),
                "source": "metamath_" + ty,
            }
        )
    print({k: got[k] for k in quota}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_stage2.jsonl")
    ap.add_argument("--stage1", default="data/sft.jsonl")
    ap.add_argument("--n-omi2", type=int, default=55000)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    used_pairs = set()
    for l in open(args.stage1):
        r = json.loads(l)
        used_pairs.add((r["question"], r["completion"].rsplit("\n\nANSWER:", 1)[0].strip()))
    print(f"[stage2] {len(used_pairs)} (problem, solution) pairs already used", flush=True)

    pool = omi2_fresh(used_pairs, args.n_omi2)
    pool += metamath_gsm(
        {"GSM_Rephrased": 26000, "GSM_AnsAug": 14000, "GSM_FOBAR": 7000, "GSM_SV": 7000}
    )
    human = gsm8k_train_rows()
    pool += human

    rng.shuffle(pool)
    print(f"[stage2] pool {len(pool)}", flush=True)

    n_fs = int(len(pool) * args.fewshot_frac)
    ks = [1, 2, 3, 4, 5, 8, 10]
    with open(args.out, "w") as f:
        for i, r in enumerate(pool):
            system = fewshot_block(rng.sample(human, rng.choice(ks))) if i < n_fs else None
            f.write(
                json.dumps(
                    {
                        "system": system,
                        "user": MATH_PROMPT_TEMPLATE.format(prompt=r["question"]),
                        "completion": f"{r['solution']}\n\nANSWER: {r['answer']}",
                        "question": r["question"],
                        "answer": r["answer"],
                        "source": r["source"],
                    }
                )
                + "\n"
            )
    print(f"[stage2] wrote {len(pool)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
