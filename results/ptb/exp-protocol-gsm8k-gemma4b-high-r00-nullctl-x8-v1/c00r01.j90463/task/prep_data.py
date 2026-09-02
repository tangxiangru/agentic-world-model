import json, random, re
from datasets import load_dataset

random.seed(0)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ds = load_dataset("openai/gsm8k", "main")["train"]

def clean(ans):
    body, target = ans.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return body, target.strip().replace(",", "")

recs = []
for r in ds:
    body, target = clean(r["answer"])
    recs.append({"question": r["question"].strip(), "reasoning": body, "target": target})

random.shuffle(recs)

def fewshot_block(shots):
    return "\n\n".join(
        f"{s['question']}\n\nReasoning:\n{s['reasoning']}\n\nANSWER: {s['target']}" for s in shots
    )

out = []
n = len(recs)
for i, r in enumerate(recs):
    system = None
    # ~12% of samples get a few-shot system prefix (eval uses a 10-shot system message)
    if i % 8 == 0:
        k = random.choice([2, 3, 4])
        pool = [recs[(i + j * 137 + 11) % n] for j in range(1, k + 1)]
        system = fewshot_block(pool)
    out.append({
        "system": system,
        "user": MATH_PROMPT_TEMPLATE.format(prompt=r["question"]),
        "assistant": f"{r['reasoning']}\n\nANSWER: {r['target']}",
    })

with open("train_data.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")

# decontamination check input: raw question + solution text
with open("decon_input.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps({"text": r["question"] + "\n" + r["reasoning"] + "\nANSWER: " + r["target"]}) + "\n")

print("wrote", len(out), "examples")
