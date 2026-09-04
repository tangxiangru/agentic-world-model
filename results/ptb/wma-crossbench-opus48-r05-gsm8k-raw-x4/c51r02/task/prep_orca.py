import json, re
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")

def extract_last_number(text):
    # prefer a number near the end, strip $ and commas
    matches = NUM_RE.findall(text)
    if not matches:
        return None
    val = matches[-1]
    val = val.replace("$", "").replace(",", "")
    val = val.rstrip(".")
    if val in ("", "-", "."):
        return None
    return val

ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
rows = []
for r in ds:
    q = r["question"].strip()
    a = r["answer"].strip()
    if len(a) < 15 or len(a) > 2000:
        continue
    tgt = extract_last_number(a)
    if tgt is None:
        continue
    # skip if answer text is trivially short / no reasoning
    prompt = MATH_PROMPT_TEMPLATE.format(prompt=q)
    completion = f"{a}\n\nANSWER: {tgt}"
    rows.append({"prompt": prompt, "completion": completion, "target": tgt})

print("kept", len(rows), "of", len(ds))
with open("train_orca_full.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
# also write a check file
with open("check_orca.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"text": r["prompt"] + "\n" + r["completion"]}) + "\n")
print("sample:")
print(rows[0]["completion"][:400])
