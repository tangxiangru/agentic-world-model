#!/usr/bin/env python3
"""Pull gsm8k-derived rows from a LARGER OpenMathInstruct-2 split and keep only problems
no earlier card has trained on, so the next stage sees new problems rather than second
solutions to old ones. Still GSM8K-train-derived; the test split is never read."""
import argparse, json, random, sys
from datasets import load_dataset
from prep_data import PROMPT_TEMPLATE, clean_solution, is_plain_number

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train_5M")
ap.add_argument("--seen", default="data/sft_gsm8k.jsonl")
ap.add_argument("--n", type=int, default=26000)
ap.add_argument("--max-chars", type=int, default=3000)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--out", default="data/new_problems.jsonl")
a = ap.parse_args()

seen = set()
for line in open(a.seen):
    seen.add(json.loads(line)["question"].strip())
print("seen problems:", len(seen), flush=True)

ds = load_dataset("nvidia/OpenMathInstruct-2", split=a.split)
ds = ds.filter(lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8)
print("gsm8k-derived rows in", a.split, ":", len(ds), flush=True)

rng = random.Random(a.seed)
taken, rows = set(), []
order = list(range(len(ds)))
rng.shuffle(order)
for i in order:
    r = ds[i]
    q = r["problem"].strip()
    if q in seen or q in taken:
        continue
    ans = r["expected_answer"].strip()
    if not is_plain_number(ans):
        continue
    sol = clean_solution(r["generated_solution"])
    if not sol or len(sol) > a.max_chars:
        continue
    ans = ans.replace(",", "")
    target = f"{sol}\n\nANSWER: {ans}"
    if target.count("ANSWER:") != 1:
        continue
    taken.add(q)
    rows.append({"prompt": PROMPT_TEMPLATE.format(prompt=q), "completion": target,
                 "question": q, "answer": ans, "source": "omi2_new_problem"})
    if len(rows) >= a.n:
        break
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "rows with unseen problems to", a.out)
