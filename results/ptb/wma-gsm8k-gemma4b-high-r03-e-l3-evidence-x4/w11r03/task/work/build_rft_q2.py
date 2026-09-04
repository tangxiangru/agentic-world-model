import json, random, hashlib, re
from datasets import load_dataset
NUM=re.compile(r"-?\$?\d[\d,]*\.?\d*")
def norm(s):
    s=str(s).strip().replace(',','').replace('$','').rstrip('.')
    try: f=float(s)
    except: return None
    return int(f) if f==int(f) else round(f,6)
used={hashlib.md5(json.loads(l)['question'].strip().lower().encode()).hexdigest() for l in open('data/sft_pool.jsonl')}
dev={json.loads(l)['question'].strip().lower() for l in open('data/dev300_gsm8ktrain.jsonl')}
existing=[json.loads(l) for l in open('data/rft_questions.jsonl')]
for e in existing: used.add(hashlib.md5(e['question'].strip().lower().encode()).hexdigest())
out=[]
mm=load_dataset("meta-math/MetaMathQA",split="train")
idx=list(range(len(mm))); random.Random(3).shuffle(idx)
n=0
for i in idx:
    if n>=22000: break
    r=mm[i]
    if not r['type'].startswith('GSM'): continue
    q=r['query'].strip(); h=hashlib.md5(q.lower().encode()).hexdigest()
    if h in used or q.lower() in dev: continue
    m=re.search(r"The answer is:?\s*(.+)$", r['response'].strip())
    if not m: continue
    g=norm(m.group(1))
    if g is None: continue
    used.add(h); out.append({"id":f"mm-{i}","question":q,"gold":str(g)}); n+=1
print("metamath GSM:",n,flush=True)
orca=load_dataset("microsoft/orca-math-word-problems-200k",split="train")
oi=list(range(len(orca))); random.Random(5).shuffle(oi)
m2=0
for i in oi:
    if m2>=15000: break
    r=orca[i]; q=r['question'].strip(); h=hashlib.md5(q.lower().encode()).hexdigest()
    if h in used or q.lower() in dev: continue
    ms=NUM.findall(r['answer'])
    if not ms: continue
    g=norm(ms[-1])
    if g is None: continue
    used.add(h); out.append({"id":f"orca-{i}","question":q,"gold":str(g)}); m2+=1
print("orca fresh:",m2,flush=True)
with open('data/rft_questions_extra.jsonl','w') as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("total extra",len(out))
