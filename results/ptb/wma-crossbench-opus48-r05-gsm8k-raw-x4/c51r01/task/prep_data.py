#!/usr/bin/env python3
"""Build SFT data for GSM8K, formatted to match the eval exactly.

Eval user prompt (0-shot form) uses MATH_PROMPT_TEMPLATE and expects the model
to produce step-by-step reasoning ending with a final line "ANSWER: <num>".
We store conversational prompt-completion JSONL so TRL applies the gemma chat
template (identical to templates/gemma3.jinja).
"""
import json, re, os, random

random.seed(0)
os.makedirs("data", exist_ok=True)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
HASH = re.compile(r"\s*####.*", re.MULTILINE)

def clean_reasoning(s):
    s = CALC.sub("", s)
    s = HASH.sub("", s)
    return s.strip()

def clean_num(s):
    s = s.strip()
    s = s.replace("$", "").replace(",", "").strip()
    s = s.rstrip(".")
    # keep leading minus, digits, decimal point
    m = re.search(r"-?\d[\d]*\.?\d*", s)
    return m.group(0) if m else s

def make_record(question, reasoning, target):
    user = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    assistant = reasoning.strip() + "\n\nANSWER: " + target
    return {
        "prompt": [{"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": assistant}],
    }, (question.strip() + "\n" + reasoning.strip())

def dump(records, texts, name):
    with open(f"data/{name}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(f"data/{name}.txt", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")  # one json-escaped line per doc
    print(f"{name}: {len(records)} records")

from datasets import load_dataset

# ---------------- GSM8K train ----------------
gsm = load_dataset("openai/gsm8k", "main", split="train")
recs, txts = [], []
for ex in gsm:
    ans = ex["answer"]
    target = ans.split("####")[-1].strip()
    target = clean_num(target)
    reasoning = ans.split("####")[0]
    reasoning = clean_reasoning(reasoning)
    r, t = make_record(ex["question"], reasoning, target)
    recs.append(r); txts.append(t)
dump(recs, txts, "gsm_train")

# ---------------- MetaMathQA GSM-derived ----------------
mm = load_dataset("meta-math/MetaMathQA", split="train")
GSM_TYPES = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"}
by_type = {t: [] for t in GSM_TYPES}
for ex in mm:
    if ex["type"] in GSM_TYPES:
        by_type[ex["type"]].append(ex)

PER_TYPE = 15000
recs, txts = [], []
skipped = 0
for t, items in by_type.items():
    random.shuffle(items)
    taken = 0
    for ex in items:
        if taken >= PER_TYPE:
            break
        resp = ex["response"]
        if "The answer is:" not in resp:
            skipped += 1; continue
        head, tail = resp.rsplit("The answer is:", 1)
        target = clean_num(tail)
        if not re.search(r"\d", target):
            skipped += 1; continue
        reasoning = clean_reasoning(head)
        if len(reasoning) < 5:
            skipped += 1; continue
        r, tx = make_record(ex["query"], reasoning, target)
        recs.append(r); txts.append(tx)
        taken += 1
    print(f"  {t}: took {taken} of {len(items)}")
print("skipped", skipped)
random.shuffle(recs)
# keep parallel text order not critical for checker (sampled subset check)
dump(recs, [t for t in txts], "metamath_gsm")
print("done")
