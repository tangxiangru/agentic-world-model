from datasets import load_dataset
import json, os
ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
print(ds)
print(ds[0])
from collections import Counter
print(Counter(ds['problem_source']))
ds.save_to_disk("/home/ben/task/data/omi2_1M")
