import glob, json, re, collections
import pyarrow.parquet as pq
from datasets import load_dataset

used = set()
for l in open('data/sft_v2.jsonl'):
    used.add(json.loads(l)['question'].strip())

NUMRE = re.compile(r"^-?\d{1,12}(,\d{3})*(\.\d+)?$")
def clean(a):
    a=a.strip().replace(',','').replace('$','').replace('%','')
    if a.endswith('.0'): a=a[:-2]
    return a

pool=[]
d=load_dataset('openai/gsm8k','main')['train']
for r in d:
    pool.append({'question':r['question'].strip(),'answer':clean(r['answer'].split('####')[1]),'src':'gsm8k_train'})

seen=set(x['question'] for x in pool)
files=sorted(glob.glob('/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet'))
naug=0
for f in files:
    t=pq.read_table(f)
    P=t.column('problem').to_pylist(); A=t.column('expected_answer').to_pylist(); S=t.column('problem_source').to_pylist()
    for p,a,s in zip(P,A,S):
        if s!='augmented_gsm8k': continue
        p=p.strip(); a=clean(a)
        if p in seen or p in used: continue
        if not NUMRE.match(a): continue
        if len(p)>1200: continue
        seen.add(p); pool.append({'question':p,'answer':a,'src':'aug_gsm8k'}); naug+=1
        if naug>=26000: break
    if naug>=26000: break
print(collections.Counter(x['src'] for x in pool))
with open('data/rft_pool.jsonl','w') as f:
    for r in pool: f.write(json.dumps(r)+'\n')
print('total',len(pool))
