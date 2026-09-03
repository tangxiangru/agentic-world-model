from datasets import load_dataset
d = load_dataset("openai/gsm8k", "main")
print(d)
d["train"].to_json("/home/ben/task/data/gsm8k_train_raw.jsonl")
print("done")
