#!/usr/bin/env python3
"""Build SFT rows from microsoft/orca-math-word-problems-200k.

These are 200k grade-school word problems that are NOT derived from GSM8K,
which is the only pool of genuinely new problems left after OpenMathInstruct-2's
gsm8k slice is exhausted.  There is no gold-answer field, so the last number in
the worked solution is taken as the answer and the row is dropped when that is
not an unambiguous integer.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq

from common_fmt import STOP_TOKEN

ORCA = "/home/ben/hf_cache/hub/datasets--microsoft--orca-math-word-problems-200k/snapshots/*/data/*.parquet"
GSM = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"
NUM = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")
CALC = re.compile(r"<<[^>]*>>")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=45000)
ap.add_argument("--max-chars", type=int, default=2200)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--fewshot-frac-4", type=float, default=0.10)
ap.add_argument("--fewshot-frac-10", type=float, default=0.10)
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(ORCA)):
    for r in pq.read_table(f).to_pylist():
        q, ans = (r["question"] or "").strip(), (r["answer"] or "").strip()
        if not q or not ans or len(ans) > a.max_chars or len(ans) < 40:
            continue
        if "\\boxed" in ans or "####" in ans or "ANSWER:" in ans:
            continue
        nums = NUM.findall(ans)
        if not nums:
            continue
        v = nums[-1].replace(",", "")
        # the final number must be an unambiguous integer and must be the last
        # thing the solution says (allowing a trailing period / unit-free tail)
        if not re.match(r"^-?\d+$", v):
            continue
        tail = ans[ans.rfind(nums[-1]) + len(nums[-1]):]
        if len(tail.strip(" .!\n")) > 12:
            continue
        rows.append((q, ans, v))

print(f"orca rows passing the filters: {len(rows)}")
rng = random.Random(a.seed)
rng.shuffle(rows)
rows = rows[: a.n]

gt = pq.read_table(sorted(glob.glob(GSM))[0]).to_pylist()
pool = []
for r in gt:
    body, _, ans = r["answer"].rpartition("####")
    pool.append((r["question"].strip(), CALC.sub("", body).strip(), ans.strip().replace(",", "")))

n4 = int(len(rows) * a.fewshot_frac_4)
n10 = int(len(rows) * a.fewshot_frac_10)
with open(a.out, "w") as fo, open(a.out.replace(".jsonl", ".decon.jsonl"), "w") as fd:
    for i, (q, sol, v) in enumerate(rows):
        k = 4 if i < n4 else (10 if i < n4 + n10 else 0)
        shots = [pool[j] for j in rng.sample(range(len(pool)), k)] if k else []
        target = f"{sol}\n\nANSWER: {v}" + STOP_TOKEN
        fo.write(json.dumps({"question": q, "target": target, "answer": v,
                             "n_shot": k, "shots": shots, "source": "orca-math"}) + "\n")
        fd.write(json.dumps({"question": q, "answer": target}) + "\n")
print("wrote", a.out, len(rows))
