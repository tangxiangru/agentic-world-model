import json, random, hashlib
from datasets import load_dataset

used = set()
for l in open('data/sft_pool.jsonl'):
    used.add(hashlib.md5(json.loads(l)['question'].strip().lower().encode()).hexdigest())
dev = {json.loads(l)['question'].strip().lower() for l in open('data/dev300_gsm8ktrain.jsonl')}
out = []
gsm = load_dataset("openai/gsm8k","main",split="train")
n_g=0
for i,r in enumerate(gsm):
    q=r['question'].strip()
    if q.lower() in dev: continue
    out.append({"id":f"gsmtrain-{i}","question":q,"gold":r['answer'].rpartition('####')[2].strip()}); n_g+=1
print("gsm8k train questions:", n_g, flush=True)

omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
idx=list(range(len(omi))); random.Random(11).shuffle(idx)
n_o=0
seen=set()
for i in idx:
    if n_o>=12000: break
    r=omi[i]
    if r['problem_source'] not in ('gsm8k','augmented_gsm8k'): continue
    q=r['problem'].strip(); h=hashlib.md5(q.lower().encode()).hexdigest()
    if h in used or h in seen or q.lower() in dev: continue
    try: float(str(r['expected_answer']).replace(',',''))
    except: continue
    seen.add(h)
    out.append({"id":f"omi-{i}","question":q,"gold":str(r['expected_answer'])}); n_o+=1
print("fresh omi questions:", n_o, flush=True)
with open('data/rft_questions.jsonl','w') as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("total", len(out))
