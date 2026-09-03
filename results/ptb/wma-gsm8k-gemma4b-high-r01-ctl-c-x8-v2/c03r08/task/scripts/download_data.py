from datasets import load_dataset
import json, os

os.makedirs('/home/ben/task/data/raw', exist_ok=True)

# 1. GSM8K train
d = load_dataset('openai/gsm8k', 'main', split='train')
d.to_json('/home/ben/task/data/raw/gsm8k_train.jsonl')
print('gsm8k train', len(d), flush=True)

# 2. OpenMathInstruct-2 1M subset
try:
    o = load_dataset('nvidia/OpenMathInstruct-2', split='train_1M')
    print('omi2 1M', len(o), o.column_names, flush=True)
    o = o.filter(lambda x: 'gsm8k' in x['problem_source'], num_proc=8)
    print('omi2 gsm8k-sourced', len(o), flush=True)
    o.to_json('/home/ben/task/data/raw/omi2_gsm8k.jsonl')
except Exception as e:
    print('OMI2 FAIL', e, flush=True)

# 3. MetaMathQA
try:
    m = load_dataset('meta-math/MetaMathQA', split='train')
    print('metamath', len(m), m.column_names, flush=True)
    m.to_json('/home/ben/task/data/raw/metamath.jsonl')
except Exception as e:
    print('METAMATH FAIL', e, flush=True)
print('DONE', flush=True)
