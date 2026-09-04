from datasets import load_dataset
import collections, json
ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
print(ds)
print(json.dumps(ds[0], indent=2)[:2000])
print(collections.Counter(ds['problem_source']))
