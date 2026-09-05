import json, re
from datasets import load_dataset

# Exact prompt template used by inspect_evals/gsm8k
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def build(record):
    DELIM = "####"
    q = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer).strip()
    prompt = MATH_PROMPT_TEMPLATE.format(prompt=q)
    completion = f"{reasoning}\n\nANSWER: {target}"
    return {"prompt": prompt, "completion": completion, "target": target}

ds = load_dataset("openai/gsm8k", "main", split="train")
rows = [build(r) for r in ds]
with open("train_gsm8k.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "examples")
print("=== sample prompt ===")
print(rows[0]["prompt"])
print("=== sample completion ===")
print(rows[0]["completion"])
