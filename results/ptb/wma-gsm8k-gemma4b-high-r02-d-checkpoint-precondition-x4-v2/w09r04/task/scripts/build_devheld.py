"""Carve a held-out dev split out of the GSM8K *train* split.

These items are never used for training (see scripts/build_sft_data.py, which
reads the same seed/split and takes the complement). They exist so that failure
analysis and watch sets never touch the benchmark test copy.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data"
SEED = 1234
N_DEV = 500


def main() -> None:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    dev_idx = sorted(idx[:N_DEV])
    train_idx = sorted(idx[N_DEV:])

    OUT.mkdir(exist_ok=True)
    with (OUT / "gsm8k_devheld.jsonl").open("w") as f:
        for i in dev_idx:
            r = ds[i]
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"id": f"trh-{i}", "question": r["question"],
                                "gold": gold, "answer": r["answer"]}) + "\n")
    with (OUT / "gsm8k_trainpool.jsonl").open("w") as f:
        for i in train_idx:
            r = ds[i]
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"id": f"tr-{i}", "question": r["question"],
                                "gold": gold, "answer": r["answer"]}) + "\n")
    print(f"dev {len(dev_idx)}  trainpool {len(train_idx)}")


if __name__ == "__main__":
    main()
