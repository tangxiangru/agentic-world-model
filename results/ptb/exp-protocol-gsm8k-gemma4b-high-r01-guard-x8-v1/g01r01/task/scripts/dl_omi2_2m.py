from datasets import load_dataset
ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_2M")
g = ds.filter(lambda r: r['problem_source'] in ('gsm8k','augmented_gsm8k'), num_proc=8)
print(len(g))
g.save_to_disk("/home/ben/task/data/omi2_gsm_2M")
print("done")
