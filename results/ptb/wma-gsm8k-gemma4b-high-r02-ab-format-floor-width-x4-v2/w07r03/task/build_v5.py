#!/usr/bin/env python3
"""Reproduce data/sft_pool_v5.jsonl (the exp-05 corpus).

Stage 1 (seed 31): 50 000 augmented-GSM solutions absent from sft_pool.jsonl and
sft_pool_v4.jsonl + up to 40 000 integer-answer MATH rows on problems absent from
sft_pool_v4.jsonl + 14 000 gsm8k-source rows.
Stage 2 (seed 41): trim to 46 000 / 32 000 / 14 000 = 92 000 rows for the budget.
"""
import collections, json, pickle, random, re, sys
sys.path.insert(0, ".")
from build_data import MARKER, STOP, fewshot_block, render_prompt, strip_boxed

rng = random.Random(31)
heldout = {json.loads(l)["question"].strip() for l in open("data/heldout_dev300.jsonl")}
from datasets import load_dataset
tr = load_dataset("openai/gsm8k", "main")["train"]
pool = [(r["question"].strip(), r["answer"].split("####")[0].strip(),
         r["answer"].split("####")[1].strip())
        for r in tr if r["question"].strip() not in heldout]

used, usedq = set(), set()
for f in ("data/sft_pool.jsonl", "data/sft_pool_v4.jsonl"):
    for l in open(f):
        used.add(json.loads(l)["completion"])
for l in open("data/sft_pool_v4.jsonl"):
    r = json.loads(l)
    usedq.add(r["prompt"].split("answer to the problem.\n\n", 1)[1]
              .split("\n\nRemember to put", 1)[0].strip())

v1 = [json.loads(l) for l in open("data/sft_pool.jsonl")]
v2 = [json.loads(l) for l in open("data/sft_pool_v2.jsonl")]
freshg = [r for r in v2 if r["completion"] not in used]; rng.shuffle(freshg)
origs = [r for r in v1 if r["src"] == "gsm8k"]; rng.shuffle(origs)

d = pickle.load(open("data/omi2_math_int_byprob.pkl", "rb"))
keys = [k for k in d if k.strip() not in usedq]; rng.shuffle(keys)
mrows = []
for q in keys:
    if len(mrows) >= 40000:
        break
    body, ans, src = d[q][0]
    txt = strip_boxed(body).strip()
    if "ANSWER:" in txt.upper():
        continue
    comp = f"{txt}\n\n{MARKER}{ans}{STOP}"
    nums = re.findall(r"-?\d+(?:\.\d+)?", comp.replace(",", ""))
    if not nums or nums[-1] != ans.replace(",", "") or len(comp) > 4000 or comp in used:
        continue
    fs = fewshot_block(rng.sample(pool, rng.randint(2, 10))) if rng.random() < 0.2 else None
    mrows.append({"prompt": render_prompt(q.strip(), fs), "completion": comp,
                  "answer": ans, "src": src, "n_shot": 0 if fs is None else 1})

mix = freshg[:50000] + mrows + origs[:14000]
rng.shuffle(mix)

rng2 = random.Random(41)
by = collections.defaultdict(list)
for r in mix:
    by[r["src"]].append(r)
for k in by:
    rng2.shuffle(by[k])
final = (by["gsm8k"][:14000] + by["augmented_gsm8k"][:46000]
         + (by["augmented_math"] + by["math"])[:32000])
rng2.shuffle(final)
with open("data/sft_pool_v5.jsonl", "w") as f:
    for r in final:
        f.write(json.dumps(r) + "\n")
print(len(final), collections.Counter(r["src"] for r in final))
