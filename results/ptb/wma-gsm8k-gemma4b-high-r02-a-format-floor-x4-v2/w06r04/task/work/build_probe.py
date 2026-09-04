"""Build the held-out probe set: the LAST 200 rows of the gsm8k TRAIN split.

These 200 rows are excluded from every training file this session (see
work/build_data.py, which stops at index 7273). No test item is touched.
"""
import json
import os

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from datasets import load_dataset

HELD_OUT_FROM = 7273  # train rows [7273:] are the probe set

ds = load_dataset("openai/gsm8k", "main")["train"]
rows = []
for i in range(HELD_OUT_FROM, len(ds)):
    r = ds[i]
    gold = r["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({"id": f"train-{i}", "question": r["question"], "gold": gold})

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_set.jsonl")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} probe rows to {out}")
