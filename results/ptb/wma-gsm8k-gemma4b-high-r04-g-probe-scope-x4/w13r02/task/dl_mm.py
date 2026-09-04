from datasets import load_dataset
import json, collections
ds = load_dataset("meta-math/MetaMathQA", split="train")
print(ds)
print(collections.Counter(ds["type"]))
for i in [0, 100000, 300000]:
    print(json.dumps(ds[i], indent=1)[:1200])
