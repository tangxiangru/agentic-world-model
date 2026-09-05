"""Convert normalized JSONL (tools, query, answer) into tokenized training tensors
matching the exact eval chat-template format. Prompt tokens are masked (-100)."""
import json, sys, os
from bfcl_evaluation_code import create_tool_info_from_dict
from inspect_ai.model._openai import openai_chat_tool_param
from transformers import AutoTokenizer

SNAP='/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d'

def get_tok():
    tok=AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template=open('templates/gemma3_tool_calling.jinja').read()
    return tok

def to_oai(tools):
    infos=[create_tool_info_from_dict(t) for t in tools]
    out=[]
    for i in infos:
        p=openai_chat_tool_param(i)
        if hasattr(p,'model_dump'):
            p=p.model_dump(exclude_none=True)
        out.append(p)
    return out

def build_example(tok, rec, max_len=2048):
    tools=to_oai(rec['tools'])
    user=[{"role":"user","content":rec['query']}]
    asst={"role":"assistant","content":"","tool_calls":[
        {"id":"c0","type":"function","function":{"name":rec['answer']['name'],"arguments":rec['answer']['arguments']}}
    ]}
    prompt=tok.apply_chat_template(user, tools=tools, add_generation_prompt=True, tokenize=False)
    full=tok.apply_chat_template(user+[asst], tools=tools, add_generation_prompt=False, tokenize=False)
    if not full.startswith(prompt.rstrip('\n')) and not full.startswith(prompt):
        # fall back to common-prefix handling below
        pass
    pids=tok(prompt, add_special_tokens=False)['input_ids']
    fids=tok(full, add_special_tokens=False)['input_ids']
    # common prefix length
    L=0
    for a,b in zip(pids,fids):
        if a==b: L+=1
        else: break
    if len(fids)>max_len:
        return None
    labels=[-100]*L + fids[L:]
    assert len(labels)==len(fids)
    if all(x==-100 for x in labels):
        return None
    return {"input_ids":fids,"labels":labels}

def main():
    inputs=sys.argv[1].split(',')
    out=sys.argv[2]
    max_len=int(sys.argv[3]) if len(sys.argv)>3 else 2048
    tok=get_tok()
    n=0; skipped=0
    with open(out,'w') as o:
        for path in inputs:
            for line in open(path):
                line=line.strip()
                if not line: continue
                rec=json.loads(line)
                try:
                    ex=build_example(tok,rec,max_len)
                except Exception as e:
                    skipped+=1; continue
                if ex is None:
                    skipped+=1; continue
                o.write(json.dumps(ex)+'\n')
                n+=1
    print(f"wrote {n} examples, skipped {skipped} to {out}")

if __name__=='__main__':
    main()
