import sys, json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
model=sys.argv[1]
tok=AutoTokenizer.from_pretrained(model)
ct=open("templates/gemma3.jinja").read()
MP=('Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n{prompt}\n\nRemember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\nReasoning:')
test=load_dataset("openai/gsm8k","main",split="test")
q=test[0]["question"]
# zero-shot
p0=tok.apply_chat_template([{"role":"user","content":MP.format(prompt=q)}],tokenize=False,add_generation_prompt=True,chat_template=ct)
# few-shot: build 3-shot system like eval
train=load_dataset("openai/gsm8k","main",split="train")
def fs(ex):
    a=ex["answer"].split("####"); t=a[-1].strip(); r="####".join(a[:-1]).strip()
    return f"{ex['question']}\n\nReasoning:\n{r}\n\nANSWER: {t}"
sysmsg="\n\n".join(fs(train[i]) for i in range(3))
p1=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":MP.format(prompt=q)}],tokenize=False,add_generation_prompt=True,chat_template=ct)
llm=LLM(model=model,gpu_memory_utilization=0.85,max_model_len=4096,dtype="bfloat16")
sp=SamplingParams(temperature=0.0,max_tokens=512)  # no stop override; rely on eos
for name,p in [("ZERO-SHOT",p0),("FEW-SHOT(3)",p1)]:
    o=llm.generate([p],sp)[0].outputs[0]
    txt=o.text
    print(f"\n===== {name} finish_reason={o.finish_reason} len_tokens={len(o.token_ids)} n_ANSWER={txt.count('ANSWER:')} =====")
    print(txt[:400])
