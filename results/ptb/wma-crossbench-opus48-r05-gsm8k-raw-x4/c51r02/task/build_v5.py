import json, random, re, collections
random.seed(3)

def clean(comp):
    m=re.search(r"ANSWER:\s*([^\n]+)",comp)
    if not m: return None
    ans=m.group(1).strip(); body=comp[:m.start()].rstrip()
    if len(body)<5: return None
    full=f"{body}\n\nANSWER: {ans}"
    if full.count("ANSWER:")!=1 or "\nmodel\n" in full: return None
    return full

byq=collections.defaultdict(list)
for fn in ["star_raw.jsonl","star2_raw.jsonl"]:
    for line in open(fn):
        r=json.loads(line); c=clean(r["completion"])
        if c: byq[r["prompt"]].append(c)
star=[]
for p,cs in byq.items():
    u=list(dict.fromkeys(cs)); u.sort(key=len)
    for c in u[:3]:
        star.append({"prompt":p,"completion":c})
print("combined STaR:",len(star),"from",len(byq),"q")

mix=[json.loads(l) for l in open("train_mix.jsonl")]
fewshot=[r for r in mix if r["prompt"].count("Reasoning:")>1]
single=[r for r in mix if r["prompt"].count("Reasoning:")==1]
gsm_gold=[r for r in single if "<<" in r["completion"]]
orca=[r for r in single if "<<" not in r["completion"]]
seen=set(); gsm1=[]
for r in gsm_gold:
    if r["prompt"] in seen: continue
    seen.add(r["prompt"]); gsm1.append(r)
random.shuffle(orca); random.shuffle(fewshot)
out=star+gsm1+orca[:22000]+fewshot[:3000]
random.shuffle(out)
with open("train_v5.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("TOTAL v5:",len(out),"| star",len(star),"gsm",len(gsm1),"orca 22k fewshot 3k")
