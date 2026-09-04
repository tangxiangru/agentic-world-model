import argparse, json, re, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

ap=argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dev", default="/home/ben/task/work/dev_probe.jsonl")
ap.add_argument("--template", default="/home/ben/task/templates/gemma3_tool_calling.jinja")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", default=None)
a=ap.parse_args()

tok=AutoTokenizer.from_pretrained(a.model); tok.chat_template=open(a.template).read()
tok.padding_side="left"
m=AutoModelForImageTextToText.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation="eager").cuda().eval()

rows=[json.loads(l) for l in open(a.dev)][:a.n]
TC=re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def canon(name, args, tools):
    tool=None
    for t in tools:
        fn=t.get("function",t)
        if fn["name"]==name: tool=fn; break
    if tool is None: return args
    params=tool.get("parameters",{}) or {}
    req=set(params.get("required") or [])
    props=params.get("properties") or {}
    defaults={k:v["default"] for k,v in props.items() if k not in req and isinstance(v,dict) and "default" in v}
    return {**defaults, **args}

def parse_out(text):
    mt=TC.search(text)
    if not mt: return None,None
    try:
        obj=json.loads(mt.group(1))
        return obj.get("name"), obj.get("arguments",{})
    except Exception:
        return None,None

correct=0; parse_fail=0; results=[]
for i in range(0,len(rows),a.bs):
    batch=rows[i:i+a.bs]
    prompts=[tok.apply_chat_template([{"role":"user","content":r["query"]}],tools=r["tools"],
                                     tokenize=False,add_generation_prompt=True) for r in batch]
    enc=tok(prompts,add_special_tokens=False,return_tensors="pt",padding=True).to("cuda")
    with torch.no_grad():
        out=m.generate(**enc,max_new_tokens=256,do_sample=False,pad_token_id=tok.pad_token_id)
    gen=out[:,enc["input_ids"].shape[1]:]
    for r,g in zip(batch,gen):
        text=tok.decode(g,skip_special_tokens=False)
        name,args=parse_out(text)
        if name is None: parse_fail+=1
        ok=False
        if name==r["gold_name"]:
            try: ok=canon(name,args or {},r["tools"])==canon(r["gold_name"],r["gold_args"],r["tools"])
            except Exception: ok=False
        correct+=int(ok)
        results.append({"q":r["query"][:80],"gold":r["gold_name"],"pred":name,"ok":ok})
acc=correct/len(rows)
print(f"dev_probe acc {acc:.4f} ({correct}/{len(rows)}) parse_fail {parse_fail}")
if a.out:
    json.dump({"acc":acc,"correct":correct,"n":len(rows),"parse_fail":parse_fail,"results":results}, open(a.out,"w"), indent=1)
