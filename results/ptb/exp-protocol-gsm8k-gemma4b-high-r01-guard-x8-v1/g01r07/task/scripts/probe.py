"""Greedy accuracy on a held-out set, with and without the grader's 10-shot prefix."""
import argparse, json, re, sys
from transformers import AutoTokenizer
SNAP="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MPT="""
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
def norm(s):
    s=str(s).strip().replace(",","").replace("$","").rstrip(".")
    try: return f"{float(s):.5g}"
    except ValueError: return None
ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--data",default="data/heldout400.jsonl")
ap.add_argument("--n",type=int,default=400); ap.add_argument("--out",default=None); ap.add_argument("--gpu-frac",type=float,default=0.85)
a=ap.parse_args()
tok=AutoTokenizer.from_pretrained(SNAP); tok.chat_template=open("templates/gemma3.jinja").read()
sysmsg=open("data/eval_system_message.txt").read()
rows=[json.loads(l) for l in open(a.data)][:a.n]
def build(q,kshot):
    m=([{"role":"system","content":sysmsg}] if kshot else [])+[{"role":"user","content":MPT.replace("{prompt}",q)}]
    return tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
from vllm import LLM, SamplingParams
llm=LLM(model=a.model,gpu_memory_utilization=a.gpu_frac,max_model_len=4096,dtype="bfloat16",seed=0)
sp=SamplingParams(n=1,temperature=0.0,max_tokens=768,stop_token_ids=[1,106])
res={}
for kshot in (True,False):
    outs=llm.generate([build(r["question"],kshot) for r in rows],sp)
    ok=0; nostop=0; lens=[]
    for r,o in zip(rows,outs):
        t=o.outputs[0].text; lens.append(len(t))
        if o.outputs[0].finish_reason=="length": nostop+=1
        m=re.findall(r"ANSWER:\s*([^\n]*)",t)
        if m and norm(m[-1])==norm(r["answer"]): ok+=1
    res["10shot" if kshot else "0shot"]={"acc":ok/len(rows),"n":len(rows),"no_stop":nostop,"mean_chars":sum(lens)/len(lens)}
    print(("10shot" if kshot else "0shot"), res["10shot" if kshot else "0shot"], flush=True)
if a.out: json.dump(res,open(a.out,"w"),indent=2)
