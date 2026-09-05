import json,sys
from transformers import AutoTokenizer
M="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
tmpl=open("templates/gemma3.jinja").read()
tok=AutoTokenizer.from_pretrained(M)
path=sys.argv[1]
P=[];C=[]
for l in open(path):
    d=json.loads(l)
    m=d["messages"]
    p=tok.apply_chat_template(m[:-1],chat_template=tmpl,tokenize=False,add_generation_prompt=True)
    P.append(p); C.append(m[-1]["content"].strip()+"<end_of_turn>")
pi=tok(P,add_special_tokens=False)["input_ids"]; ci=tok(C,add_special_tokens=False)["input_ids"]
tot=sorted(len(a)+len(b) for a,b in zip(pi,ci))
n=len(tot)
import statistics
print("n",n,"mean",sum(tot)/n,"p50",tot[n//2],"p95",tot[int(n*.95)],"p99",tot[int(n*.99)],"max",tot[-1])
for L in (768,1024,1280,1536,2048):
    print(L, "frac over:", sum(1 for t in tot if t>L)/n)
print("total tokens", sum(tot))
