#!/usr/bin/env python3
"""Expanded pool: the same gsm8k-derived slice of OpenMathInstruct-2, but from
the full 14M-row train split instead of the 1M subsample, so the count of
DISTINCT problems grows rather than just the count of solutions per problem.

Same cleaning contract as scripts/build_data.py: numeric answers only,
\\boxed{X} rewritten to X in place, one "ANSWER: <n>" line appended last,
terminator attached.  The GSM8K test split is never read.
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

TASK = "/home/ben/task"
GLOB = ("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
        "snapshots/*/data/train-000*-of-00032.parquet")
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
EOT = "<end_of_turn>"

import sys
sys.path.insert(0, f"{TASK}/scripts")
from build_data import clean_solution  # same cleaner, byte-for-byte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{TASK}/data/pool2.jsonl")
    ap.add_argument("--per-problem", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    prompt_tpl = json.load(open(f"{TASK}/data/eval_prompt.json"))["prompt_template"]
    dev_q = {json.loads(l)["question"]
             for l in open(f"{TASK}/data/dev_internal.jsonl")}

    best: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    files = sorted(glob.glob(GLOB))
    assert len(files) == 32, f"{len(files)} shards found"
    for n, f in enumerate(files):
        t = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"])
        src = t.column("problem_source").to_pylist()
        prob = t.column("problem").to_pylist()
        sol = t.column("generated_solution").to_pylist()
        ans = t.column("expected_answer").to_pylist()
        for s, p, so, an in zip(src, prob, sol, ans):
            if s not in ("gsm8k", "augmented_gsm8k"):
                continue
            an = (an or "").strip()
            if not NUMERIC.match(an):
                continue
            p = p.strip()
            if p in dev_q:
                continue
            cands = best[p]
            if len(cands) >= a.per_problem and len(so) >= cands[-1][0]:
                continue          # already have enough shorter solutions
            body = clean_solution(so, an)
            if body is None or "ANSWER:" in so:
                continue
            cands.append((len(so), body, an))
            cands.sort(key=lambda c: c[0])
            del cands[a.per_problem:]
        print(f"  {n+1}/32 {os.path.basename(f)}: {len(best)} distinct problems",
              flush=True)

    rng = random.Random(a.seed)
    rows = []
    for q, cands in best.items():
        for _, body, an in cands:
            rows.append({"question": q,
                         "prompt": prompt_tpl.replace("{prompt}", q),
                         "target": body + EOT, "answer": an,
                         "src": "omi2:train"})
    rng.shuffle(rows)

    kept = 0
    with open(a.out, "w") as fh:
        for r in rows:
            if r["target"].count("ANSWER:") != 1:
                continue
            if r["target"].rsplit("ANSWER:", 1)[1].strip() != r["answer"] + EOT:
                continue
            fh.write(json.dumps(r) + "\n")
            kept += 1
    print(f"wrote {kept} rows over {len(best)} distinct problems to {a.out}")


if __name__ == "__main__":
    main()
