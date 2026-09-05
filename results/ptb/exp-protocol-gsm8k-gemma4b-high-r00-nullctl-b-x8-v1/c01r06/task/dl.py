from datasets import load_dataset
import traceback
for name, kw in [
    ("nvidia/OpenMathInstruct-2", dict(split="train_1M")),
    ("microsoft/orca-math-word-problems-200k", dict(split="train")),
]:
    try:
        d = load_dataset(name, **kw)
        print("OK", name, d)
        print(d[0])
    except Exception as e:
        traceback.print_exc()
