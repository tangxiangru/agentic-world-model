from datasets import load_dataset
o = load_dataset('nvidia/OpenMathInstruct-2', split='train_2M')
print('omi2 2M', len(o), flush=True)
o = o.filter(lambda x: 'gsm8k' in x['problem_source'], num_proc=8)
print('gsm8k-sourced', len(o), flush=True)
o.to_json('/home/ben/task/data/raw/omi2_gsm8k_2m.jsonl')
print('DONE', flush=True)
