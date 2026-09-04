"""Is llm.generate() double-prefixing <bos>?  The gemma3 template already emits
one; vLLM tokenises raw text prompts with add_special_tokens=True."""
import json, sys
sys.path.insert(0,"work")
from rft_sample import PROMPT_TEMPLATE, last_number, norm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

M="ckpts/exp-04/final"
tok=AutoTokenizer.from_pretrained(M)
tpl=open("templates/gemma3.jinja").read()
items=[json.loads(l) for l in open("data/rft_questions.jsonl")][:300]
texts=[tok.apply_chat_template([{"role":"user","content":PROMPT_TEMPLATE.format(prompt=it["question"])}],
        chat_template=tpl, tokenize=False, add_generation_prompt=True) for it in items]
ids=[tok(t, add_special_tokens=False).input_ids for t in texts]
print("raw-string path first ids:", tok(texts[0]).input_ids[:4])
print("explicit ids first ids   :", ids[0][:4])
llm=LLM(model=M, gpu_memory_utilization=0.9, dtype="bfloat16", max_model_len=2048)
sp=SamplingParams(temperature=0.0, max_tokens=512)
for name, inp in [("raw_str(add_special=True)", texts),
                  ("token_ids(no extra bos)", [TokensPrompt(prompt_token_ids=i) for i in ids])]:
    outs=llm.generate(inp, sp)
    c=sum(1 for it,o in zip(items,outs) if last_number(o.outputs[0].text)==norm(it["gold"]))
    print(f"{name}: {c}/{len(items)} = {c/len(items):.3f}")
