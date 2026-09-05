import json, re, random
random.seed(0)
rows=[json.loads(l) for l in open('clean_pool.jsonl')]
INSTR_PREFIXES = [
    "",  # plain
]
def ok(r):
    resp=r['response']; ins=r['instruction']
    if len(ins)<15 or len(ins)>3000: return False
    if len(resp)<20 or len(resp)>4000: return False
    # require a python code block or clear python code
    if '```python' not in resp and 'def ' not in resp: return False
    return True
rows=[r for r in rows if ok(r)]
# prefer examples whose response contains ```python fenced block
def has_block(r): return '```python' in r['response']
blocked=[r for r in rows if has_block(r)]
other=[r for r in rows if not has_block(r)]
print("blocked",len(blocked),"other",len(other))
# For 'other' (no fence), wrap the code in a python fence if it looks like code-only
def wrap(r):
    resp=r['response'].strip()
    if '```' in resp:  # has some fence already, keep
        return resp
    return "```python\n"+resp+"\n```"
final=[]
for r in blocked:
    final.append({"instruction":r['instruction'].strip(),"response":r['response'].strip()})
for r in other:
    final.append({"instruction":r['instruction'].strip(),"response":wrap(r)})
random.shuffle(final)
with open('train_final.jsonl','w') as f:
    for r in final: f.write(json.dumps(r)+"\n")
print("final",len(final))
# length stats via char proxy
import statistics
L=[len(r['instruction'])+len(r['response']) for r in final]
print("char len mean",int(statistics.mean(L)),"p90",sorted(L)[int(len(L)*0.9)])
