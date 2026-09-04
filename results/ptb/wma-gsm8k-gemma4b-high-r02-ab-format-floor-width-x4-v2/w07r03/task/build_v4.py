#!/usr/bin/env python3
"""Reproduce data/sft_pool_v4.jsonl exactly (the exp-03 corpus).

Two stages, matching how the file on disk was produced:
  stage 1 (seed 11): 60 000 integer-answer MATH rows + 62 000 fresh
                     augmented-GSM solutions + all 28 339 gsm8k rows
  stage 2 (seed 23): trim to 34 000 MATH + 52 000 augmented-GSM + 28 339 gsm8k
                     = 114 339 rows, to fit the wall-clock budget

Inputs: data/omi2_math_int_byprob.pkl (built by the math extraction pass over
OpenMathInstruct-2), data/sft_pool.jsonl (exp-02's corpus), data/sft_pool_v2.jsonl.
"""
import collections, json, pickle, random, re, sys

sys.path.insert(0, ".")
from build_data import MARKER, STOP, fewshot_block, render_prompt, strip_boxed

rng = random.Random(11)
heldout = {json.loads(l)["question"].strip() for l in open("data/heldout_dev300.jsonl")}

from datasets import load_dataset
tr = load_dataset("openai/gsm8k", "main")["train"]
pool = [(r["question"].strip(),
         r["answer"].split("####")[0].strip(),
         r["answer"].split("####")[1].strip())
        for r in tr if r["question"].strip() not in heldout]

# ---- stage 1 -------------------------------------------------------------
d = pickle.load(open("data/omi2_math_int_byprob.pkl", "rb"))
keys = list(d.keys())
rng.shuffle(keys)
math_rows = []
for q in keys:
    if len(math_rows) >= 60000:
        break
    body, ans, src = d[q][0]
    txt = strip_boxed(body).strip()
    if "ANSWER:" in txt.upper():
        continue
    comp = f"{txt}\n\n{MARKER}{ans}{STOP}"
    nums = re.findall(r"-?\d+(?:\.\d+)?", comp.replace(",", ""))
    if not nums or nums[-1] != ans.replace(",", ""):
        continue
    if len(comp) > 4000:
        continue
    math_rows.append({"question": q.strip(), "completion": comp,
                      "answer": ans, "src": src})

v1 = [json.loads(l) for l in open("data/sft_pool.jsonl")]
v2 = [json.loads(l) for l in open("data/sft_pool_v2.jsonl")]
seen = {r["completion"] for r in v1}
orig = [r for r in v1 if r["src"] == "gsm8k"]
fresh = [r for r in v2 if r["completion"] not in seen]
rng.shuffle(fresh)

mix = orig + fresh[:62000] + [dict(r, prompt=None) for r in math_rows]
rng.shuffle(mix)
out = []
for r in mix:
    if r.get("prompt"):                      # already rendered in v1/v2
        out.append({k: r[k] for k in ("prompt", "completion", "answer", "src", "n_shot")})
    else:
        fs = fewshot_block(rng.sample(pool, rng.randint(2, 10))) if rng.random() < 0.2 else None
        out.append({"prompt": render_prompt(r["question"], fs), "completion": r["completion"],
                    "answer": r["answer"], "src": r["src"], "n_shot": 0 if fs is None else 1})

# ---- stage 2 -------------------------------------------------------------
rng2 = random.Random(23)
by = collections.defaultdict(list)
for r in out:
    by[r["src"]].append(r)
for k in by:
    rng2.shuffle(by[k])
final = by["gsm8k"] + by["augmented_gsm8k"][:52000] + (by["augmented_math"] + by["math"])[:34000]
rng2.shuffle(final)

with open("data/sft_pool_v4.jsonl", "w") as f:
    for r in final:
        f.write(json.dumps(r) + "\n")
print(len(final), collections.Counter(r["src"] for r in final),
      "fewshot", round(sum(r["n_shot"] > 0 for r in final) / len(final), 3))
