"""Hold out a dev set from the GSM8K *train* split (test split is never touched)."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TASK_DIR, gsm8k_gold, fewshot_system_message  # noqa: E402

from datasets import load_dataset

d = load_dataset("openai/gsm8k", "main")["train"]
_, fewshot_qs = fewshot_system_message()
fewshot_set = set(fewshot_qs)

idx = list(range(len(d)))
random.Random(0).shuffle(idx)

dev, dev_ids = [], set()
for i in idx:
    r = d[i]
    if r["question"] in fewshot_set:
        continue
    dev.append({"id": f"train-{i}", "question": r["question"], "gold": gsm8k_gold(r["answer"])})
    dev_ids.add(i)
    if len(dev) == 500:
        break

os.makedirs(os.path.join(TASK_DIR, "data"), exist_ok=True)
with open(os.path.join(TASK_DIR, "data", "dev500.jsonl"), "w") as f:
    for r in dev:
        f.write(json.dumps(r) + "\n")
with open(os.path.join(TASK_DIR, "data", "dev500_train_idx.json"), "w") as f:
    json.dump(sorted(dev_ids), f)

# the dev questions, verbatim, so the data builder can exclude near-duplicates
with open(os.path.join(TASK_DIR, "data", "dev500_questions.json"), "w") as f:
    json.dump([r["question"] for r in dev], f)

print(f"dev500: {len(dev)} items, held out from gsm8k train")
print(dev[0])
