#!/usr/bin/env python3
"""Build SFT dataset for GSM8K, matching the eval output format.

Output: train_data.jsonl with fields {"question": ..., "response": ...}
The response always ends with a line 'ANSWER: <number>'.
"""
import re, json, random
from datasets import load_dataset

random.seed(0)

def clean_gsm_reasoning(ans: str):
    # ans like: "...<<48/2=24>>24 clips...\n#### 72"
    parts = ans.split("####")
    reasoning = parts[0].strip()
    target = parts[1].strip()
    # strip calculator annotations <<...>>
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    return reasoning, target

def extract_num(s):
    s = s.strip()
    # remove commas, $, trailing period
    s = s.replace(",", "").replace("$", "").strip()
    return s

out = []

# 1) GSM8K train split
gsm = load_dataset("openai/gsm8k", "main", split="train")
for r in gsm:
    q = r["question"].strip()
    reasoning, target = clean_gsm_reasoning(r["answer"])
    target = extract_num(target)
    resp = f"{reasoning}\nANSWER: {target}"
    out.append({"question": q, "response": resp, "src": "gsm8k_train"})

n_gsm = len(out)
print("gsm8k_train:", n_gsm)

# 2) MetaMathQA GSM-derived augmentations
meta = load_dataset("meta-math/MetaMathQA", split="train")
ans_re = re.compile(r"The answer is:\s*(.+)\s*$")
kept_meta = 0
per_type_cap = {"GSM_AnsAug": 100000, "GSM_Rephrased": 100000, "GSM_FOBAR": 100000, "GSM_SV": 100000}
type_counts = {}
for r in meta:
    t = r["type"]
    if not t.startswith("GSM"):
        continue
    if type_counts.get(t, 0) >= per_type_cap.get(t, 0):
        continue
    resp = r["response"].strip()
    m = ans_re.search(resp)
    if not m:
        continue
    num = extract_num(m.group(1))
    # keep purely numeric answers (GSM answers are numbers)
    if not re.fullmatch(r"-?\d+(\.\d+)?", num):
        continue
    # remove trailing '#### X' and 'The answer is: X' lines from the body
    body = resp
    body = re.sub(r"\n?####\s*.*?$", "", body, flags=re.DOTALL)  # careful
    # better: cut at first '####' or 'The answer is:'
    idx_candidates = [i for i in [resp.find("\n####"), resp.find("The answer is:")] if i != -1]
    cut = min(idx_candidates) if idx_candidates else len(resp)
    body = resp[:cut].strip()
    if len(body) < 10:
        continue
    q = r["query"].strip()
    new_resp = f"{body}\nANSWER: {num}"
    out.append({"question": q, "response": new_resp, "src": t})
    type_counts[t] = type_counts.get(t, 0) + 1
    kept_meta += 1

print("meta types:", type_counts, "total meta:", kept_meta)

random.shuffle(out)
with open("train_data_full.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("TOTAL:", len(out))

# also write a plain-text version (question only) for contamination check on questions
with open("questions_only.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps({"text": r["question"]}) + "\n")
