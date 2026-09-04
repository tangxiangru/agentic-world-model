import argparse, json, re, os
from datasets import load_dataset
from vllm import LLM, SamplingParams

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--gpu", type=float, default=0.85)
    return p.parse_args()
args=parse()

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def build_prompt(q):
    return f"<bos><start_of_turn>user\n{MATH_PROMPT_TEMPLATE.format(prompt=q)}<end_of_turn>\n<start_of_turn>model\n"

def norm_num(s):
    s=s.replace(",","").replace("$","").strip().rstrip(".").rstrip("%").strip()
    m=re.match(r"^-?\d+\.?\d*$", s)
    if not m:
        mm=re.search(r"-?\d+\.?\d*", s)
        s=mm.group(0) if mm else s
    try:
        f=float(s); return str(int(f)) if f==int(f) else str(f)
    except: return None

def extract_answer(text):
    # first ANSWER: occurrence (model should stop after it)
    m=re.search(r"ANSWER:\s*([^\n]+)", text)
    if not m: return None
    return norm_num(m.group(1))

gsm=load_dataset("openai/gsm8k","main",split="train")
data=[(r["question"], r["answer"].split("####")[-1].strip()) for r in gsm]
if args.limit>0: data=data[:args.limit]

llm=LLM(model=args.model, gpu_memory_utilization=args.gpu, max_model_len=1536, dtype="bfloat16")
sp=SamplingParams(n=args.n, temperature=args.temp, top_p=0.95, max_tokens=args.max_tokens,
                  stop=["<end_of_turn>"])

prompts=[build_prompt(q) for q,_ in data]
outs=llm.generate(prompts, sp)

kept=0; seen=set(); fout=open(args.out,"w")
n_solved=0
for (q,gold), o in zip(data, outs):
    goldn=norm_num(gold)
    got_correct=False
    for comp in o.outputs:
        txt=comp.text.strip()
        ans=extract_answer(txt)
        if ans is not None and goldn is not None and ans==goldn:
            got_correct=True
            key=(q, txt[:80])
            if key in seen: continue
            seen.add(key)
            # ensure it ends cleanly with ANSWER line
            idx=txt.rfind("ANSWER:")
            clean=txt[:idx].rstrip()+"\n\nANSWER: "+gold
            fout.write(json.dumps({"prompt": MATH_PROMPT_TEMPLATE.format(prompt=q),
                                   "completion": clean})+"\n")
            kept+=1
    if got_correct: n_solved+=1
fout.close()
print(f"questions: {len(data)} | solved(>=1 correct): {n_solved} ({100*n_solved/len(data):.1f}%) | kept solutions: {kept}")
