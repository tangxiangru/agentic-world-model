import json, sys
sys.path.insert(0,"scripts")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm_gen import render, last_number, eq, fewshot_prefix
from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE
M="/home/ben/task/ckpts/exp-02/final"
rows=[json.loads(l) for l in open("data/dev_train300.jsonl")][:100]
tok=AutoTokenizer.from_pretrained(M)
print("tokenizer eos:", tok.eos_token, tok.eos_token_id, flush=True)
pre=fewshot_prefix(10)+"\n\n"
print("fewshot built", flush=True)
llm=LLM(model=M, gpu_memory_utilization=0.85, max_model_len=4096, seed=0)
print("llm up", flush=True)
for tag, prefix in [("fewshot10", pre), ("zeroshot","")]:
    prompts=[TokensPrompt(prompt_token_ids=tok(render(prefix+MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip())), add_special_tokens=False)["input_ids"]) for r in rows]
    sp=SamplingParams(temperature=0.0, max_tokens=768, stop_token_ids=[1,106], skip_special_tokens=False)
    outs=llm.generate(prompts, sp)
    acc=sum(eq(last_number(o.outputs[0].text), str(r["gold"])) for r,o in zip(rows,outs))/len(rows)
    ends=sum(o.outputs[0].text.rstrip().endswith("<end_of_turn>") for o in outs)/len(outs)
    print(f"[{tag}] acc={acc:.3f} ends_with_eot={ends:.2f}", flush=True)
    print("SAMPLE:", repr(outs[0].outputs[0].text[-200:]), flush=True)
