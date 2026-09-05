#!/usr/bin/env python3
"""Build a mixture concentrated on REAL GSM8K-train problems.

exp-02..exp-05 all trained on a mixture ~80% OpenMathInstruct-2 *augmented* gsm8k
(synthetic variants). The graded items are real GSM8K. This build inverts that
ratio: every row is a solution to one of the 7473 real gsm8k train problems,
except a replay slice that keeps the few-shot share and the augmented style alive.
"""
import glob, json, os, random, sys
from collections import Counter

sys.path.insert(0, "scripts")
import fmt
import pyarrow.parquet as pq

rng = random.Random(11)
REPLAY_FS = int(os.environ.get("REPLAY_FS", 2500))
REPLAY_ZS = int(os.environ.get("REPLAY_ZS", 9500))
OUT = os.environ.get("GOLD_OUT", "data/sft_v4_gold.jsonl")
rows = []

# 1. human-written CoT for all 7473 real train problems
gsm_q = {}
for p in glob.glob("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"):
    for r in pq.read_table(p).to_pylist():
        reasoning, ans = fmt.clean_gsm8k_reasoning(r["answer"])
        q = r["question"].strip()
        gsm_q[q] = ans
        rows.append({"question": q, "reasoning": reasoning, "answer": ans, "source": "gsm8k_human"})

# 2. OMI-2 solutions to those same real problems (problem_source == "gsm8k"), up to 4 each
seen = Counter()
for p in sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-*.parquet")):
    for r in pq.read_table(p, columns=["problem", "generated_solution", "expected_answer", "problem_source"]).to_pylist():
        if r["problem_source"] != "gsm8k":
            continue
        q = r["problem"].strip()
        if seen[q] >= 4:
            continue
        sol = fmt.clean_omi_solution(r["generated_solution"])
        if "ANSWER:" in sol or "\\boxed" in sol or not (40 <= len(sol) <= 4000):
            continue
        seen[q] += 1
        rows.append({"question": q, "reasoning": sol,
                     "answer": fmt.normalize_number(r["expected_answer"]), "source": "omi2_gsm8k_real"})

out = []
shot_pool = [(r["question"], r["reasoning"], r["answer"]) for r in rows if r["source"] == "gsm8k_human"]
for r in rows:
    shots = None
    if rng.random() < 0.13:
        k = rng.choice([1, 2, 3, 4, 8, 10])
        shots = [s for s in rng.sample(shot_pool, k) if s[0] != r["question"]]
    out.append({"prompt": fmt.render_prompt(r["question"], shots),
                "target": fmt.build_target(r["reasoning"], r["answer"]),
                "question": r["question"], "answer": r["answer"],
                "source": r["source"], "n_shots": len(shots) if shots else 0})

# 3. self-generated correct chains on the same real problems (already built, exp-04's data)
out += [json.loads(l) for l in open("data/rft_gsm.jsonl")]

# 4. replay so the augmented style and the k-shot share survive
v2 = [json.loads(l) for l in open("data/sft_v2.jsonl")]
fs = [r for r in v2 if r.get("n_shots", 0) > 0]
zs = [r for r in v2 if r.get("n_shots", 0) == 0]
rng.shuffle(fs); rng.shuffle(zs)
out += fs[:REPLAY_FS] + zs[:REPLAY_ZS]

rng.shuffle(out)
with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
nfs = sum(1 for r in out if r.get("n_shots", 0) > 0)
print(f"rows={len(out)} few_shot={nfs} ({nfs/len(out):.2%})")
print(Counter(r["source"] for r in out))
real = sum(1 for r in out if r["source"] in ("gsm8k_human", "omi2_gsm8k_real", "rft_gsm8k_train"))
print(f"rows on real gsm8k train problems: {real} ({real/len(out):.1%})")
