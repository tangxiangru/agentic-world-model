"""Train-side problem pool for rejection sampling: {question, answer}."""
import json, random
from datasets import load_dataset
rng = random.Random(0)
rows=[]
d = load_dataset("openai/gsm8k","main",split="train")
for r in d:
    rows.append({"question": r["question"].strip(), "answer": r["answer"].split("####")[-1].strip().replace(",",""), "src":"gsm8k_train"})
import re
NUM=re.compile(r"^-?\d[\d,]*(\.\d+)?$")
seen=set(r["question"].lower()[:200] for r in rows)
aug=[]
with open("data/omi2_gsm.jsonl") as f:
    for line in f:
        r=json.loads(line)
        if r["problem_source"]!="augmented_gsm8k": continue
        a=str(r["expected_answer"]).strip().replace(",","")
        if not NUM.match(a): continue
        k=r["problem"].strip().lower()[:200]
        if k in seen: continue
        seen.add(k)
        aug.append({"question": r["problem"].strip(), "answer": a, "src":"aug_gsm8k"})
rng.shuffle(aug)
rows += aug[:30000]
rng.shuffle(rows)
with open("data/rft_pool.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print(len(rows), "problems")
