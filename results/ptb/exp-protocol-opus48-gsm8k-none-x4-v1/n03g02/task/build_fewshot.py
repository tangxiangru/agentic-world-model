#!/usr/bin/env python3
"""Wrap base training examples with few-shot context (matching eval format) to teach
the model to answer only the final question and STOP. Fewshots = raw GSM8K train
(with <<>> calc annotations, exactly like the eval's sample_to_fewshot)."""
import os, re, json, random
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from datasets import load_dataset
random.seed(123)

base_file = "combined_rft_union.jsonl"
out_file = "combined_fewshot.jsonl"

# Build fewshot pool from raw GSM8K train (question, raw_reasoning, target) -- like eval
ds = load_dataset("openai/gsm8k", "main")["train"]
pool = []
for ex in ds:
    parts = ex["answer"].split("####")
    reasoning = parts[0].strip()
    target = parts[1].strip().replace(",", "")
    pool.append((ex["question"].strip(), reasoning, target))

def fewshot_block(q, r, t):
    return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {t}"

# read base examples
base = [json.loads(l) for l in open(base_file)]
print("base examples:", len(base), "| fewshot pool:", len(pool))

n = 0
with open(out_file, "w") as f:
    for rec in base:
        user = rec["prompt"][0]["content"]
        assistant = rec["completion"][0]["content"]
        # sample k fewshots (vary 2..6) not equal to this question text
        k = random.choice([2, 3, 4, 5, 6])
        shots = random.sample(pool, k)
        fs = "\n\n".join(fewshot_block(*s) for s in shots)
        msgs = {
            "prompt": [
                {"role": "system", "content": fs},
                {"role": "user", "content": user},
            ],
            "completion": [{"role": "assistant", "content": assistant}],
        }
        f.write(json.dumps(msgs) + "\n")
        n += 1
print("wrote", n, "few-shot-wrapped examples to", out_file)
