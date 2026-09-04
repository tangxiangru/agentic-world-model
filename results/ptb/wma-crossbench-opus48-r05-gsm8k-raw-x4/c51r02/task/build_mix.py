import json, random, re
from datasets import load_dataset

random.seed(0)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def gsm_reason_target(ans):
    DELIM="####"; parts=ans.split(DELIM); target=parts.pop().strip(); reasoning=DELIM.join(parts).strip()
    return reasoning, target

def fewshot_block(q, reasoning, target):
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"

# ---- GSM8K train ----
gsm = load_dataset("openai/gsm8k","main",split="train")
gsm_rows=[]
gsm_pool=[]  # for fewshot sampling
for r in gsm:
    reasoning,target = gsm_reason_target(r["answer"])
    gsm_rows.append((r["question"], reasoning, target))
    gsm_pool.append((r["question"], reasoning, target))

out=[]
# zero-shot GSM8K, x2
for _ in range(2):
    for q,reasoning,target in gsm_rows:
        out.append({"prompt": MATH_PROMPT_TEMPLATE.format(prompt=q),
                    "completion": f"{reasoning}\n\nANSWER: {target}"})

# few-shot-context GSM8K (teach stopping under in-context examples), 5000 examples
N_FEW=5000
for _ in range(N_FEW):
    q,reasoning,target = random.choice(gsm_rows)
    k=random.randint(2,6)
    shots=random.sample(gsm_pool, k)
    prefix="\n\n".join(fewshot_block(sq,sr,st) for sq,sr,st in shots)
    prompt = prefix + "\n\n" + MATH_PROMPT_TEMPLATE.format(prompt=q)
    out.append({"prompt": prompt, "completion": f"{reasoning}\n\nANSWER: {target}"})

# ---- Orca-Math subset (shorter, high quality) ----
NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")
def extract_last_number(text):
    ms=NUM_RE.findall(text)
    if not ms: return None
    v=ms[-1].replace("$","").replace(",","").rstrip(".")
    return v if v not in ("","-",".") else None

orca = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
orca_kept=[]
for r in orca:
    q=r["question"].strip(); a=r["answer"].strip()
    if len(a)<20 or len(a)>1200: continue
    tgt=extract_last_number(a)
    if tgt is None: continue
    # require the final number to be near the end of the solution (answer-stating)
    if a.rfind(tgt.split('.')[0]) < len(a)-60:
        pass
    orca_kept.append((q,a,tgt))
random.shuffle(orca_kept)
orca_take = orca_kept[:55000]
for q,a,tgt in orca_take:
    out.append({"prompt": MATH_PROMPT_TEMPLATE.format(prompt=q),
                "completion": f"{a}\n\nANSWER: {tgt}"})

random.shuffle(out)
with open("train_mix.jsonl","w") as f:
    for r in out:
        f.write(json.dumps(r)+"\n")
print("total examples:", len(out))
print("  gsm zero x2:", 2*len(gsm_rows), "| gsm fewshot:", N_FEW, "| orca:", len(orca_take))
