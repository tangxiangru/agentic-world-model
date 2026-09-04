from datasets import load_dataset
import json, collections
ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
print(ds)
print(collections.Counter(ds["problem_source"]))
print(json.dumps(ds[0], indent=1)[:2000])
