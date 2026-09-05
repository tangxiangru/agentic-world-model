"""The OpenMathInstruct-2 GSM8K rows that neither sft_v1 nor sft_v2 used."""
import hashlib, json, sys
from datasets import load_dataset
sys.path.insert(0,"work")
from build_sft_data import make_row
STOP="<end_of_turn>"
seen=set()
for fn in ("data/sft_v1.jsonl","data/sft_v2.jsonl"):
    for l in open(fn):
        d=json.loads(l)
        seen.add(hashlib.md5((d['question']+"||"+d['target'].replace(STOP,'')).lower().encode()).hexdigest())
dev={json.loads(l)['question'].strip().lower() for l in open('data/dev300_gsm8ktrain.jsonl')}
print("seen pairs:",len(seen),flush=True)
omi=load_dataset("nvidia/OpenMathInstruct-2",split="train_1M")
out=[]
for i in range(len(omi)):
    r=omi[i]
    if r['problem_source'] not in ('gsm8k','augmented_gsm8k'): continue
    q=r['problem'].strip()
    if q.lower() in dev: continue
    m=make_row(q,r['generated_solution'],r['expected_answer'])
    if m is None: continue
    h=hashlib.md5((q+"||"+m[1]).lower().encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    out.append({"prompt":m[0],"target":m[1]+STOP,"source":"omi2_gsm8k_rest","question":q,"fewshot":0})
with open('data/omi2_rest.jsonl','w') as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("wrote",len(out))
