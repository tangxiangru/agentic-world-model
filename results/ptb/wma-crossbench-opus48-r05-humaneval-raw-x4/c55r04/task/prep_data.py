import os, json, re, random
os.environ["HF_HOME"]="/home/ben/hf_cache"
from datasets import load_dataset
random.seed(0)

out = []

def add(instruction, response):
    instruction = instruction.strip()
    response = response.strip()
    if not instruction or not response:
        return
    out.append({"instruction": instruction, "response": response})

# 1. Magicoder OSS-Instruct (problem/solution), python-focused
print("loading magicoder oss-instruct")
ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
for r in ds:
    if r.get("lang","").lower() != "python":
        continue
    add(r["problem"], r["solution"])
print("after oss:", len(out))

# 2. Magicoder Evol-Instruct (instruction/response)
print("loading magicoder evol")
ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
for r in ds:
    instr = r["instruction"]
    resp = r["response"]
    # keep python-ish
    if "python" in instr.lower() or "```python" in resp.lower() or "def " in resp:
        add(instr, resp)
print("after evol:", len(out))

random.shuffle(out)
with open("raw_pool.jsonl","w") as f:
    for r in out:
        f.write(json.dumps(r)+"\n")
print("total raw:", len(out))
