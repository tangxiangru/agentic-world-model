import json, os, random, argparse
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

SNAP="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TMPL="/home/ben/task/templates/gemma3_tool_calling.jinja"

ap=argparse.ArgumentParser()
ap.add_argument("--maxlen", type=int, default=2048)
ap.add_argument("--out", default="/home/ben/task/work/train.jsonl")
ap.add_argument("--decontam", default="/home/ben/task/work/decontam.jsonl")
ap.add_argument("--jsonl", default="/home/ben/task/work/train_preview.jsonl")
ap.add_argument("--dev", default="/home/ben/task/work/dev_probe.jsonl")
ap.add_argument("--devsize", type=int, default=300)
ap.add_argument("--limit", type=int, default=0)
args=ap.parse_args()

tok=AutoTokenizer.from_pretrained(SNAP)
tok.chat_template=open(TMPL).read()

ds=load_dataset("minpeter/xlam-function-calling-60k-parsed", split="train")
print("raw rows", len(ds))

def norm_tools(tools):
    # already list of {"type":"function","function":{...}}
    out=[]
    for t in tools:
        if isinstance(t, str):
            t=json.loads(t)
        if "function" not in t:
            t={"type":"function","function":t}
        out.append(t)
    return out

records=[]
decontam=[]
skip_multi=0; skip_shape=0; skip_len=0
lengths=[]
for row in ds:
    msgs=row["messages"]; tools=row["tools"]
    if isinstance(tools, str): tools=json.loads(tools)
    if not tools: skip_shape+=1; continue
    # need: first user, then assistant with exactly 1 tool_call
    if len(msgs)<2: skip_shape+=1; continue
    if msgs[0]["role"]!="user": skip_shape+=1; continue
    asst=msgs[1]
    if asst["role"]!="assistant" or not asst.get("tool_calls"): skip_shape+=1; continue
    tcs=asst["tool_calls"]
    if len(tcs)!=1: skip_multi+=1; continue
    tc=tcs[0]
    fn=tc["function"] if "function" in tc else tc
    name=fn["name"]; arguments=fn["arguments"]
    if isinstance(arguments,str):
        try: arg_obj=json.loads(arguments)
        except Exception: skip_shape+=1; continue
    else:
        arg_obj=arguments
    tools=norm_tools(tools)
    query=msgs[0]["content"]
    build_msgs=[{"role":"user","content":query},
                {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":name,"arguments":json.dumps(arg_obj)}}]}]
    try:
        full=tok.apply_chat_template(build_msgs, tools=tools, tokenize=False, add_generation_prompt=False)
        prompt=tok.apply_chat_template(build_msgs[:1], tools=tools, tokenize=False, add_generation_prompt=True)
    except Exception:
        skip_shape+=1; continue
    if not full.startswith(prompt): skip_shape+=1; continue
    completion=full[len(prompt):]
    full_ids=tok(full, add_special_tokens=False)["input_ids"]
    prompt_ids=tok(prompt, add_special_tokens=False)["input_ids"]
    n=len(prompt_ids)
    # ensure token prefix; if not, back off to common prefix
    if full_ids[:n]!=prompt_ids:
        m=0
        for a,b in zip(full_ids,prompt_ids):
            if a==b: m+=1
            else: break
        n=m
    if full_ids[-1]!=106: skip_shape+=1; continue  # must end with <end_of_turn>
    lengths.append(len(full_ids))
    if len(full_ids)>args.maxlen: skip_len+=1; continue
    records.append({"prompt":prompt,"completion":completion,
                    "query":query,"tools":tools,"gold_name":name,"gold_args":arg_obj})
    decontam.append({"text": query + "\n" + name + "(" + ", ".join(f"{k}={v}" for k,v in arg_obj.items()) + ")"})
    if args.limit and len(records)>=args.limit: break

import numpy as np
L=np.array(lengths)
print(f"kept {len(records)} | skip_multi {skip_multi} skip_shape {skip_shape} skip_len {skip_len}")
print(f"token len p50 {np.percentile(L,50):.0f} p95 {np.percentile(L,95):.0f} p99 {np.percentile(L,99):.0f} max {L.max()}")
print(f"trunc share > {args.maxlen}: {(L>args.maxlen).mean()*100:.2f}%")

random.seed(0); random.shuffle(records)
os.makedirs(os.path.dirname(args.decontam), exist_ok=True)
with open(args.decontam,"w") as f:
    for d in decontam: f.write(json.dumps(d)+"\n")

dev=records[:args.devsize]
train=records[args.devsize:]
with open(args.dev,"w") as f:
    for r in dev:
        f.write(json.dumps({"query":r["query"],"tools":r["tools"],
                            "gold_name":r["gold_name"],"gold_args":r["gold_args"]})+"\n")
with open(args.out,"w") as f:
    for r in train:
        f.write(json.dumps({"prompt":r["prompt"],"completion":r["completion"]})+"\n")
print("saved train", len(train), "to", args.out, "| dev", len(dev), "to", args.dev)
