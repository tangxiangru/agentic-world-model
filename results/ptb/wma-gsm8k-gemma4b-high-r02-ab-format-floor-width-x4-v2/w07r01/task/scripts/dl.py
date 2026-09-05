import os, sys
from datasets import load_dataset
os.environ.setdefault("HF_HOME","/home/ben/hf_cache")
which = sys.argv[1]
if which == "gsm8k":
    d = load_dataset("openai/gsm8k","main")
    print("gsm8k", {k:len(v) for k,v in d.items()})
elif which == "omi2":
    d = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    print("omi2", len(d), d.column_names)
elif which == "metamath":
    d = load_dataset("meta-math/MetaMathQA", split="train")
    print("metamath", len(d), d.column_names)
