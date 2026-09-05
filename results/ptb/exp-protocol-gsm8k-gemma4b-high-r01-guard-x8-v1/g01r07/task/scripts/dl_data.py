import os, traceback
from datasets import load_dataset
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
def try_load(name, **kw):
    try:
        d = load_dataset(name, **kw)
        print("OK", name, kw, d, flush=True)
    except Exception as e:
        print("FAIL", name, kw, repr(e)[:300], flush=True)
try_load("meta-math/MetaMathQA")
try_load("nvidia/OpenMathInstruct-2", split="train_1M")
