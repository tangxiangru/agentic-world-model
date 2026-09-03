import os, sys, traceback
from datasets import load_dataset
targets = [
    ("openai/gsm8k", dict(path="openai/gsm8k", name="main", split="train")),
    ("openai/gsm8k-socratic", dict(path="openai/gsm8k", name="socratic", split="train")),
    ("nvidia/OpenMathInstruct-2", dict(path="nvidia/OpenMathInstruct-2", split="train_1M")),
    ("meta-math/MetaMathQA", dict(path="meta-math/MetaMathQA", split="train")),
]
for name, kw in targets:
    try:
        ds = load_dataset(**kw)
        print(name, len(ds), ds.column_names, flush=True)
    except Exception as e:
        print("FAIL", name, repr(e), flush=True)
        traceback.print_exc()
