import os, collections
os.environ.setdefault("HF_HOME","/home/ben/hf_cache")
from datasets import load_dataset
d = load_dataset("nvidia/OpenMathInstruct-2", split="train_5M")
print(len(d))
c = collections.Counter(d["problem_source"])
print(c)
