from datasets import load_dataset
d = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
print(d)
print(d[0])
import collections
print(collections.Counter(d["problem_source"]))
d.to_parquet("/home/ben/task/data/omi2_1M.parquet")
print("done")
