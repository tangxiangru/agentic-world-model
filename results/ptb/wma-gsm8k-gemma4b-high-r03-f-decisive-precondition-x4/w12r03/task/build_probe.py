"""Build a held-out probe set from the GSM8K *train* split.

These items are excluded from every training file, so they give a dev signal
that never touches the benchmark test set.
"""
import json
from datasets import load_dataset

PROBE_N = 250
OUT = "/home/ben/task/data/probe250.jsonl"

def main():
    train = load_dataset("openai/gsm8k", "main", split="train")
    n = len(train)
    idx = list(range(n - PROBE_N, n))  # last 250 train items
    with open(OUT, "w") as f:
        for i in idx:
            r = train[i]
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"id": f"train-{i}", "question": r["question"], "gold": gold}) + "\n")
    print(f"wrote {OUT} with {len(idx)} items (train indices {idx[0]}..{idx[-1]})")

if __name__ == "__main__":
    main()
