#!/usr/bin/env python3
"""Hold out an internal dev set of GSM8K *train* questions and remove every pool
row that shares a question with it.  The official test split is never touched.
"""
import json
import random

import datasets

N_DEV = 250
rng = random.Random(1234)

gsm = datasets.load_dataset("openai/gsm8k", "main", split="train")
idx = list(range(len(gsm)))
rng.shuffle(idx)
dev_idx = idx[:N_DEV]

dev, dev_q = [], set()
for i in dev_idx:
    r = gsm[i]
    q = r["question"].strip()
    dev_q.add(q)
    dev.append({"id": f"devtrain-{i}", "question": q,
                "gold": r["answer"].rsplit("####", 1)[1].strip()})

with open("/home/ben/task/data/dev_internal.jsonl", "w") as f:
    for d in dev:
        f.write(json.dumps(d) + "\n")

kept = dropped = 0
with open("/home/ben/task/data/pool.jsonl") as fin, \
     open("/home/ben/task/data/pool_clean.jsonl", "w") as fout:
    for line in fin:
        d = json.loads(line)
        if d["question"].strip() in dev_q:
            dropped += 1
            continue
        fout.write(line)
        kept += 1
print(f"dev={len(dev)} pool kept={kept} dropped={dropped}")
