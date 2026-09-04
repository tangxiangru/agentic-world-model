import json, random, re, collections
random.seed(2)

def clean_completion(comp):
    m = re.search(r"ANSWER:\s*([^\n]+)", comp)
    if not m: return None
    ans = m.group(1).strip()
    body = comp[:m.start()].rstrip()
    if len(body) < 5: return None
    full = f"{body}\n\nANSWER: {ans}"
    if full.count("ANSWER:") != 1: return None
    if "\nmodel\n" in full: return None
    return full

# STaR cap 2 per question
byq = collections.defaultdict(list)
with open("star_raw.jsonl") as f:
    for line in f:
        r=json.loads(line); c=clean_completion(r["completion"])
        if c: byq[r["prompt"]].append(c)
star=[]
for p,cs in byq.items():
    u=list(dict.fromkeys(cs)); u.sort(key=len)
    for c in u[:2]:
        star.append({"prompt":p,"completion":c})
print("star:",len(star))

mix=[json.loads(l) for l in open("train_mix.jsonl")]
fewshot=[r for r in mix if r["prompt"].count("Reasoning:")>1]
single=[r for r in mix if r["prompt"].count("Reasoning:")==1]
# separate gsm gold vs orca in single: gsm gold completions contain '<<' annotations typically
gsm_gold=[r for r in single if "<<" in r["completion"]]
orca=[r for r in single if "<<" not in r["completion"]]
print("fewshot",len(fewshot),"gsm_gold(dedup by prompt)",len(set(r['prompt'] for r in gsm_gold)),"orca",len(orca))

# one copy gsm gold (dedup), 20k orca, 3k fewshot
seen=set(); gsm1=[]
for r in gsm_gold:
    if r["prompt"] in seen: continue
    seen.add(r["prompt"]); gsm1.append(r)
random.shuffle(orca); random.shuffle(fewshot)
out = star + gsm1 + orca[:20000] + fewshot[:3000]
random.shuffle(out)
with open("train_v4.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("TOTAL v4:",len(out))
