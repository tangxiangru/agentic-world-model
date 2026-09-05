#!/usr/bin/env python3
"""Rebuild sft_v3_rft.jsonl with the few-shot share restored to ~13%.

The grader always prompts with 10 shots. RFT rows are generated 0-shot, so a
straight mix drops the k-shot share to 4.9%; this re-renders a slice of the RFT
rows behind a k-shot prefix (the target is still a valid answer to the same
question) and draws the replay slice stratified so both halves contribute.
"""
import glob, json, random, sys
from collections import Counter

sys.path.insert(0, "scripts")
import fmt
import pyarrow.parquet as pq

rng = random.Random(3)
rft = [json.loads(l) for l in open("data/rft_gsm.jsonl")] + [json.loads(l) for l in open("data/rft_aug.jsonl")]
rng.shuffle(rft)

shot_pool = []
for p in glob.glob("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"):
    for r in pq.read_table(p).to_pylist():
        reasoning, ans = fmt.clean_gsm8k_reasoning(r["answer"])
        shot_pool.append((r["question"].strip(), reasoning, ans))

n_fs = int(0.10 * len(rft))
for row in rft[:n_fs]:
    k = rng.choice([1, 2, 3, 4, 8, 10])
    shots = [s for s in rng.sample(shot_pool, k) if s[0] != row["question"]]
    row["prompt"] = fmt.render_prompt(row["question"], shots)
    row["n_shots"] = len(shots)

v2 = [json.loads(l) for l in open("data/sft_v2.jsonl")]
fs = [r for r in v2 if r.get("n_shots", 0) > 0]
zs = [r for r in v2 if r.get("n_shots", 0) == 0]
rng.shuffle(fs); rng.shuffle(zs)
replay = fs[:4000] + zs[:20000]

rows = rft + replay
rng.shuffle(rows)
with open("data/sft_v3_rft.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
tot_fs = sum(1 for r in rows if r.get("n_shots", 0) > 0)
ten = sum(1 for r in rows if r.get("n_shots", 0) == 10)
print(f"rows={len(rows)} few_shot={tot_fs} ({tot_fs/len(rows):.3%}) of which 10-shot={ten}")
print(Counter(r["source"] for r in rows))
