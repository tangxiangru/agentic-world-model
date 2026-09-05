#!/usr/bin/env python3
"""Third-stage data: 32k more OpenMathInstruct-2 gsm8k-derived rows whose
problems appear in neither train_v1 (exp-02) nor train_v2b (exp-04)."""
import json, random
from collections import defaultdict
from datasets import load_dataset
from build_data import CALC_RE, MATH_PROMPT_TEMPLATE, build_target, fewshot_block, normalize_answer

used = {json.loads(l)["question"] for l in open("data/train_v1.jsonl")}
used |= {json.loads(l)["question"] for l in open("data/train_v2b.jsonl")}
print("problems already used:", len(used))
rng = random.Random(11)
gsm = load_dataset("openai/gsm8k", "main", split="train")
shot_pool = []
for r in gsm:
    q = r["question"].strip(); b, _, t = r["answer"].rpartition("####")
    shot_pool.append((q, CALC_RE.sub("", b).strip(), t.strip()))
omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_2M")
omi = omi.filter(lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8)
idx = list(range(len(omi))); rng.shuffle(idx)
per = defaultdict(int); rows = []; seen = set()
for i in idx:
    if len(rows) >= 40000: break
    r = omi[i]; p = r["problem"].strip()
    if p in used or per[p] >= 2: continue
    a = normalize_answer(r["expected_answer"])
    if a is None: continue
    t = build_target(r["generated_solution"], a)
    if t is None: continue
    k = (p[:200], t[:200])
    if k in seen: continue
    seen.add(k); per[p] += 1
    rows.append({"question": p, "target": t, "answer": a, "source": r["problem_source"]})
rng.shuffle(rows)
for r in rows:
    if rng.random() < 0.25:
        k = rng.randint(1, 10); r["system"] = fewshot_block(rng.sample(shot_pool, k)); r["nshot"] = k
    else:
        r["system"], r["nshot"] = None, 0
    r["user"] = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
with open("data/train_v5.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
with open("data/train_v5_check.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
print("rows:", len(rows), "unique problems:", len(per))
