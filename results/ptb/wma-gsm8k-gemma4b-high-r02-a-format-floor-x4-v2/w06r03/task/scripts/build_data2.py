#!/usr/bin/env python3
"""Round-2 mixture: on-policy rejection-sampled rows + unseen OpenMathInstruct-2
problems, both re-rendered under the same few-shot-prefix policy as round 1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import END_OF_TURN, grade, sample_to_fewshot, user_prompt  # noqa: E402

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
OLD_SHARDS = {f"train-{i:05d}-of-00032.parquet" for i in range(4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-new-omi", type=int, default=25000)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main")["train"]
    fewshot_pool = []
    for rec in gsm:
        if "####" not in rec["answer"]:
            continue
        reasoning, target = rec["answer"].rsplit("####", 1)
        target = target.strip().replace(",", "")
        if NUM_RE.match(target):
            fewshot_pool.append((rec["question"], reasoning.strip(), target))

    # ---- on-policy rows (already rendered zero-shot by rft_sample.py)
    rows = []
    seen_q = set()
    for line in open(args.rft):
        r = json.loads(line)
        q = r["messages"][-1]["content"]
        seen_q.add(q)
        rows.append({"q_prompt": q, "body": r["completion"][: -len(END_OF_TURN)],
                     "answer": r["answer"], "src": r["src"]})
    n_rft = len(rows)

    # ---- unseen OpenMathInstruct-2 problems from shards 4..11
    import pyarrow.parquet as pq

    new = []
    per_problem = {}
    for path in sorted(glob.glob(OMI_GLOB)):
        if os.path.basename(path) in OLD_SHARDS:
            continue
        t = pq.read_table(path, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        d = t.to_pydict()
        for prob, sol, ans, src in zip(d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]):
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = (ans or "").strip().replace(",", "")
            if not NUM_RE.match(ans) or len(prob) > 2000:
                continue
            qp = user_prompt(prob)
            if qp in seen_q or per_problem.get(prob, 0) >= 1:
                continue
            body = BOXED_RE.sub(r"\1", sol or "").replace("$", "").strip()
            if not body or len(body) > 4000:
                continue
            per_problem[prob] = 1
            new.append({"q_prompt": qp, "body": body + "\n\nANSWER: " + ans,
                        "answer": ans, "src": "omi_new:" + src})
    rng.shuffle(new)
    new = new[: args.n_new_omi]
    print(f"rft rows {n_rft}, new omi rows {len(new)}", flush=True)

    rows.extend(new)
    rng.shuffle(rows)

    n_bad = n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            body = r["body"].strip()
            if not grade(body, r["answer"]):
                n_bad += 1
                continue
            messages = []
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, 10)
                messages.append({"role": "system",
                                 "content": "\n\n".join(sample_to_fewshot(*s) for s in rng.sample(fewshot_pool, k))})
                n_fs += 1
            messages.append({"role": "user", "content": r["q_prompt"]})
            f.write(json.dumps({"messages": messages, "completion": body + END_OF_TURN,
                                "answer": r["answer"], "src": r["src"]}) + "\n")
    print(f"wrote {args.out}: {len(rows)-n_bad} rows ({n_bad} dropped, {n_fs} few-shot)")


if __name__ == "__main__":
    main()
