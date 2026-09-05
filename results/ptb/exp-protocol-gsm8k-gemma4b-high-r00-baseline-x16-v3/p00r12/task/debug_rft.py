import json, random, re
from build_data import MATH_PROMPT_TEMPLATE, CALC_RE, normalize_answer, fewshot_block
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
tmpl=open("templates/gemma3.jinja").read(); M="/home/ben/task/final_model"
tok=AutoTokenizer.from_pretrained(M)
gsm=load_dataset("openai/gsm8k","main",split="train")
probs=[(r["question"].strip(), normalize_answer(r["answer"].rpartition("####")[2])) for r in list(gsm)[:300]]
prompts=[tok.apply_chat_template([{"role":"user","content":MATH_PROMPT_TEMPLATE.format(prompt=q)}],
         chat_template=tmpl,tokenize=False,add_generation_prompt=True) for q,a in probs]
llm=LLM(model=M,gpu_memory_utilization=0.85,max_model_len=3072,dtype="bfloat16")
outs=llm.generate(prompts,SamplingParams(n=3,temperature=0.8,top_p=0.95,max_tokens=512))
ANS=re.compile(r"ANSWER:\s*(\$?-?[\d,]*\.?\d+)\s*$")
import collections
c=collections.Counter(); ex=[]
for o,(q,a) in zip(outs,probs):
    for ch in o.outputs:
        t=ch.text; s=t.strip()
        m=ANS.search(s)
        fin=ch.finish_reason
        if m is None:
            c[f"no_ans_end/{fin}"]+=1
            if len(ex)<3: ex.append(("NOANS",fin,repr(s[-300:])))
        elif normalize_answer(m.group(1))!=a:
            c["wrong"]+=1
        else:
            n=s.count("ANSWER:")
            c[f"correct_ansCount={n}"]+=1
            if n!=1 and len(ex)<6: ex.append(("MULTI",n,repr(s[:200]+" ... "+s[-300:])))
print(c)
for e in ex: print(e)
