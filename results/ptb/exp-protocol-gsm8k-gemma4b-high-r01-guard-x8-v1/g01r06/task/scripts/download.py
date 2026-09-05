import os, sys
from datasets import load_dataset
os.makedirs('/home/ben/task/data/raw', exist_ok=True)

def grab(name, kw, out):
    if os.path.exists(out):
        print('skip', out, flush=True); return
    print('downloading', name, kw, flush=True)
    d = load_dataset(name, **kw)
    print(name, d, flush=True)
    d.to_json(out, lines=True)
    print('wrote', out, len(d), flush=True)

grab('openai/gsm8k', dict(name='main', split='train'), '/home/ben/task/data/raw/gsm8k_train.jsonl')
grab('microsoft/orca-math-word-problems-200k', dict(split='train'), '/home/ben/task/data/raw/orca_math.jsonl')
grab('nvidia/OpenMathInstruct-2', dict(split='train_1M'), '/home/ben/task/data/raw/omi2_1M.jsonl')
print('DONE', flush=True)
