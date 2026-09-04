import os, traceback
from datasets import load_dataset

jobs = [
    ("gsm8k",   dict(path="openai/gsm8k", name="main", split="train")),
    ("metamath",dict(path="meta-math/MetaMathQA", split="train")),
    ("orca",    dict(path="microsoft/orca-math-word-problems-200k", split="train")),
    ("omi2",    dict(path="nvidia/OpenMathInstruct-2", split="train_1M")),
]
for tag, kw in jobs:
    try:
        d = load_dataset(**kw)
        print(tag, "OK", len(d), d.column_names, flush=True)
    except Exception as e:
        print(tag, "FAIL", repr(e), flush=True)
        traceback.print_exc()
