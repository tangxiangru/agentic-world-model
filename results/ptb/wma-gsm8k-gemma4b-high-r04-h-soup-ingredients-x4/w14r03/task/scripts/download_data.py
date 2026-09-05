import os, sys, json, time
from datasets import load_dataset
t0=time.time()
out = {}
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

try:
    d = load_dataset("openai/gsm8k", "main")
    log("gsm8k", d)
except Exception as e:
    log("gsm8k FAIL", e)

for name, kwargs in [
    ("nvidia/OpenMathInstruct-2", dict(split="train_1M")),
    ("meta-math/MetaMathQA", dict(split="train")),
    ("microsoft/orca-math-word-problems-200k", dict(split="train")),
]:
    try:
        d = load_dataset(name, **kwargs)
        log(name, d)
    except Exception as e:
        log(name, "FAIL", repr(e)[:300])
log("done")
