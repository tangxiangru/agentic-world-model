"""Held-out probe set: augmented_gsm8k problems that are NOT in the training corpus."""
import json, random, re
rng=random.Random(7)
train_q=set()
for l in open("data/sft_train.jsonl"):
    train_q.add(json.loads(l)["question"].strip().lower()[:200])
NUM=re.compile(r"^-?\d[\d,]*(\.\d+)?$")
cand=[]
for l in open("data/omi2_gsm.jsonl"):
    r=json.loads(l)
    if r["problem_source"]!="augmented_gsm8k": continue
    q=r["problem"].strip()
    if q.lower()[:200] in train_q: continue
    a=str(r["expected_answer"]).strip().replace(",","")
    if not NUM.match(a): continue
    cand.append({"question":q,"answer":a})
rng.shuffle(cand)
seen=set(); out=[]
for c in cand:
    k=c["question"].lower()[:200]
    if k in seen: continue
    seen.add(k); out.append(c)
    if len(out)==400: break
with open("data/heldout400.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print(len(out),"held-out problems")
