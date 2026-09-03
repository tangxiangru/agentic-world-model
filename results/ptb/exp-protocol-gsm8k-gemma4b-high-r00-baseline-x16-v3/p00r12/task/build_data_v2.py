#!/usr/bin/env python3
"""Second-stage data: OpenMathInstruct-2 gsm8k-derived rows whose PROBLEM does
not appear in data/train_v1.jsonl, with the few-shot prefix distribution
widened to 0-10 shots (the grader's prompt carries 10)."""
import json
import random
from collections import defaultdict

from datasets import load_dataset

from build_data import (
    CALC_RE,
    MATH_PROMPT_TEMPLATE,
    build_target,
    fewshot_block,
    normalize_answer,
)

N_ROWS = 60000
used = {json.loads(l)["question"] for l in open("data/train_v1.jsonl")}
print("problems already used:", len(used))

rng = random.Random(1)
gsm = load_dataset("openai/gsm8k", "main", split="train")
shot_pool = []
for r in gsm:
    q = r["question"].strip()
    b, _, t = r["answer"].rpartition("####")
    shot_pool.append((q, CALC_RE.sub("", b).strip(), t.strip()))

omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
omi = omi.filter(lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8)
idx = list(range(len(omi)))
rng.shuffle(idx)

per = defaultdict(int)
rows, seen = [], set()
for i in idx:
    if len(rows) >= N_ROWS:
        break
    r = omi[i]
    p = r["problem"].strip()
    if p in used or per[p] >= 2:
        continue
    a = normalize_answer(r["expected_answer"])
    if a is None:
        continue
    t = build_target(r["generated_solution"], a)
    if t is None:
        continue
    k = (p[:200], t[:200])
    if k in seen:
        continue
    seen.add(k)
    per[p] += 1
    rows.append({"question": p, "target": t, "answer": a, "source": r["problem_source"]})

rng.shuffle(rows)
for r in rows:
    if rng.random() < 0.5:
        k = rng.randint(1, 10)
        r["system"] = fewshot_block(rng.sample(shot_pool, k))
        r["nshot"] = k
    else:
        r["system"], r["nshot"] = None, 0
    r["user"] = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])

with open("data/train_v2.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open("data/train_v2_check.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
print("fresh rows:", len(rows), "unique problems:", len(per))
