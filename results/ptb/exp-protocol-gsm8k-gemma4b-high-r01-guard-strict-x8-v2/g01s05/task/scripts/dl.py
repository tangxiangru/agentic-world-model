import os
os.environ.setdefault("HF_HOME","/home/ben/hf_cache")
from datasets import load_dataset
for name, kw in [("nvidia/OpenMathInstruct-2", dict(split="train_1M")),
                 ("meta-math/MetaMathQA", dict(split="train"))]:
    try:
        d = load_dataset(name, **kw)
        print(name, d, flush=True)
        print(d[0], flush=True)
    except Exception as e:
        print("FAIL", name, repr(e), flush=True)
