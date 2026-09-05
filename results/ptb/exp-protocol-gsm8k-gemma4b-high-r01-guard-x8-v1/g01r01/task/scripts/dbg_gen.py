import sys, json, re
sys.path.insert(0,'/home/ben/task/scripts')
from common import *
from gen_rft import extract, norm
from vllm import LLM, SamplingParams
tok = get_tokenizer(); sysmsg = grader_fewshot_system()
probs=[json.loads(l) for l in open('/home/ben/task/data/rft_problems_smoke.jsonl')][:150]
llm = LLM(model='/home/ben/task/ckpts/exp02/final', gpu_memory_utilization=0.85, max_model_len=3072, seed=0)
for tag, sysm in (('zeroshot',None),('fewshot',sysmsg)):
    prompts=[render_prompt(tok,p['problem'],system=sysm) for p in probs]
    sp=SamplingParams(n=4,temperature=1.0,top_p=0.95,max_tokens=640,stop_token_ids=[1, 106],seed=0)
    outs=llm.generate(prompts,sp)
    tot=cor=noext=0
    bad=[]
    for p,o in zip(probs,outs):
        g=norm(str(p['answer']))
        for c in o.outputs:
            tot+=1; e=extract(c.text.strip())
            if e is None:
                noext+=1
                if len(bad)<3: bad.append(c.text.strip()[-250:])
            elif e==g: cor+=1
    print(f"[{tag}] n={tot} correct={cor} ({cor/tot:.3f}) no_extract={noext} ({noext/tot:.3f})", flush=True)
    for b in bad: print('  BAD TAIL:', repr(b))
