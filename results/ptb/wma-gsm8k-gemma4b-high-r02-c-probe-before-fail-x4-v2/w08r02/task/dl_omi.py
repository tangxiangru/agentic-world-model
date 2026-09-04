import os
os.environ.setdefault("HF_HOME","/home/ben/hf_cache")
from datasets import load_dataset
d = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
print(d)
print(d[0])
import collections
print(collections.Counter(d["problem_source"]))
