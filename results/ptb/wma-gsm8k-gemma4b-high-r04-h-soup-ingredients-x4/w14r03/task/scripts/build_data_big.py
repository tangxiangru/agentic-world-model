#!/usr/bin/env python3
"""Same transform as build_data.py, but over OpenMathInstruct-2 train_5M."""
import json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import make_row
from datasets import load_dataset

rng = random.Random(0)
d = load_dataset("nvidia/OpenMathInstruct-2", split="train_5M")
d = d.filter(lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=6)
print("gsm8k-sourced rows:", len(d), flush=True)
per = defaultdict(list)
for r in d:
    row = make_row(r["problem"], r["generated_solution"], r["expected_answer"],
                   "openmathinstruct2:" + r["problem_source"])
    if row is not None:
        per[row["question"]].append(row)
print("distinct problems:", len(per), flush=True)
rows = []
for q, c in per.items():
    rng.shuffle(c)
    seen = set()
    for x in c[:4]:
        if x["completion"] in seen: continue
        seen.add(x["completion"]); rows.append(x)
g = load_dataset("openai/gsm8k", "main", split="train")
for r in g:
    cot, _, ans = r["answer"].rpartition("####")
    row = make_row(r["question"], cot, ans, "gsm8k:train")
    if row is not None: rows.append(row)
for r in rows:
    if not r["completion"].endswith("<end_of_turn>"):
        r["completion"] = r["completion"].rstrip() + "<end_of_turn>"
rng.shuffle(rows)
with open("/home/ben/task/data/sft_pool_5m_k4.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("wrote", len(rows), flush=True)
