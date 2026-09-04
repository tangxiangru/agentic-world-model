from datasets import load_dataset
import json
for name, kw in [("microsoft/orca-math-word-problems-200k", {}), ("meta-math/MetaMathQA", {})]:
    try:
        ds = load_dataset(name, split="train")
        print(name, ds)
        print(json.dumps(ds[0], indent=2)[:1500])
    except Exception as e:
        print(name, "FAIL", e)
