#!/usr/bin/env python3
"""Rebalance the SFT set toward multi-argument tool calls.

The base mixture skews to 0-1 arg calls (~45%), which biases the model toward
omitting arguments the query actually specifies (the dominant eval failure
mode). Keep all rich (>=2 arg) calls, oversample >=3-arg calls 2x, and keep a
capped sample of <=1-arg calls so simple-call ability is retained. Pure
subsetting + duplication of already-contamination-checked rows -- introduces no
new text. Also writes the user queries for the contamination checker.
"""
import json
import random
import re

random.seed(0)

ROWS = [json.loads(l) for l in open("data/train.jsonl")]


def nargs(c):
    m = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", c, re.S)
    if not m:
        return 0
    try:
        return len(json.loads(m.group(1)).get("arguments", {}))
    except Exception:
        return 0


def query_of(prompt):
    # last user turn content before the model generation prompt
    m = re.findall(r"<start_of_turn>user\n(.*?)<end_of_turn>", prompt, re.S)
    return m[-1].strip() if m else ""


buckets = {"le1": [], "eq2": [], "ge3": []}
for r in ROWS:
    n = nargs(r["completion"])
    if n <= 1:
        buckets["le1"].append(r)
    elif n == 2:
        buckets["eq2"].append(r)
    else:
        buckets["ge3"].append(r)

random.shuffle(buckets["le1"])
out = []
out += buckets["le1"][:4000]          # cap simple calls
out += buckets["eq2"]                  # keep all 2-arg
out += buckets["ge3"] * 2             # oversample >=3-arg 2x
random.shuffle(out)

with open("data/train_v2.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")

with open("data/train_v2_queries.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps({"question": query_of(r["prompt"])}) + "\n")

print(f"buckets: le1={len(buckets['le1'])} eq2={len(buckets['eq2'])} ge3={len(buckets['ge3'])}")
print(f"wrote data/train_v2.jsonl rows={len(out)}")
