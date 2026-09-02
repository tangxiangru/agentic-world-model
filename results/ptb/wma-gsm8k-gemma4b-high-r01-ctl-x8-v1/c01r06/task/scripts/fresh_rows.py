"""Rows from a larger built pool that were not used in a previous run."""
import argparse, json, random
ap=argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--used", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-rows", type=int, required=True)
ap.add_argument("--probe", default="/home/ben/task/data/probe200.jsonl")
ap.add_argument("--seed", type=int, default=3)
a=ap.parse_args()
used=set()
for l in open(a.used):
    used.add(json.loads(l)["completion"])
banned={json.loads(l)["question"] for l in open(a.probe)}
def qof(p):
    b=p.split("<start_of_turn>user\n",1)[1].rsplit("<end_of_turn>",1)[0]
    h="(without quotes) where $ANSWER is the answer to the problem.\n\n"
    return b.split(h,1)[-1].split("\n\nRemember to put your answer")[0].strip()
keep=[]
for l in open(a.pool):
    r=json.loads(l)
    if r["completion"] in used: continue
    if qof(r["prompt"]) in banned: continue
    keep.append(l)
print("fresh available:", len(keep))
random.Random(a.seed).shuffle(keep)
keep=keep[:a.n_rows]
open(a.out,"w").writelines(keep)
print("wrote", a.out, len(keep))
chk=a.out.replace(".jsonl","")+"_check.jsonl"
with open(a.out) as f, open(chk,"w") as o:
    for l in f:
        r=json.loads(l)
        o.write(json.dumps({"question":qof(r["prompt"]),"answer":r["completion"].replace("<end_of_turn>","").strip()})+"\n")
print("wrote", chk)
