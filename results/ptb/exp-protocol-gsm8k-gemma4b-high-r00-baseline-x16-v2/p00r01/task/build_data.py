#!/usr/bin/env python3
"""Build SFT data for gsm8k, formatted exactly as inspect_evals/gsm8k renders it."""
import json, re, random
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ds = load_dataset("openai/gsm8k", "main", split="train")

def split_rec(rec):
    q = rec["question"]
    ans = rec["answer"].split("####")
    tgt = ans.pop().strip()
    reasoning = "####".join(ans).strip()
    return q, reasoning, tgt

# ---- reproduce the eval's 10-shot system message (train split, seed 42, shuffled) ----
fs = ds.shuffle(seed=42).select(range(10))
def to_fewshot(rec):
    q, r, t = split_rec(rec)
    return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {t}"
SYSTEM = "\n\n".join(to_fewshot(r) for r in fs)
fewshot_qs = {split_rec(r)[0] for r in fs}

CALC = re.compile(r"<<[^>]*>>")
rows = []
for rec in ds:
    q, reasoning, tgt = split_rec(rec)
    if q in fewshot_qs:
        continue
    reasoning = CALC.sub("", reasoning).strip()
    tgt = tgt.replace(",", "")
    if "ANSWER:" in reasoning:
        continue
    completion = f"{reasoning}\n\nANSWER: {tgt}<end_of_turn>"
    rows.append({"prompt": MATH_PROMPT_TEMPLATE.format(prompt=q), "completion": completion})

random.Random(0).shuffle(rows)
# 20% of rows carry the exact 10-shot system message so the model sees the test-time context too
n_sys = int(0.2 * len(rows))
for i, r in enumerate(rows):
    r["system"] = SYSTEM if i < n_sys else None

with open("train_data.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open("system_prompt.txt", "w") as f:
    f.write(SYSTEM)
print("rows", len(rows), "with_system", n_sys)
print("SYSTEM tokens approx", len(SYSTEM.split()))
