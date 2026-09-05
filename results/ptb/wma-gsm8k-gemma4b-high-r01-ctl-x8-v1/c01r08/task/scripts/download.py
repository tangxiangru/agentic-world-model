from datasets import load_dataset
import traceback
for name, kw in [
    ("nvidia/OpenMathInstruct-2", dict(split="train_1M")),
    ("meta-math/MetaMathQA", dict(split="train")),
]:
    try:
        d = load_dataset(name, **kw)
        print("OK", name, d)
    except Exception as e:
        traceback.print_exc()
