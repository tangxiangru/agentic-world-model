"""Hold out a probe/dev set from the GSM8K TRAIN split (never test).

- 300 items taken from the tail of the train split.
- The 10 few-shot exemplars the grader injects (seed 42, shuffled) are excluded
  so the probe never scores an item that is also in its own prompt.
"""
import json
from datasets import load_dataset

train = load_dataset("openai/gsm8k", "main", split="train")

# reproduce the grader's few-shot selection so we can exclude those items
fewshot = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(10))
fewshot_q = {r["question"] for r in fewshot}

rows = []
for i in range(len(train) - 1, -1, -1):
    r = train[i]
    if r["question"] in fewshot_q:
        continue
    ans = r["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({"id": f"devtrain-{i}", "idx": i, "question": r["question"],
                 "gold": ans, "reasoning": r["answer"].split("####")[0].strip()})
    if len(rows) == 300:
        break
rows.reverse()

with open("data/dev_train300.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

held_out_idx = sorted(r["idx"] for r in rows)
with open("data/dev_train300_idx.json", "w") as f:
    json.dump(held_out_idx, f)
print("wrote", len(rows), "probe items; min idx", held_out_idx[0], "max idx", held_out_idx[-1])
