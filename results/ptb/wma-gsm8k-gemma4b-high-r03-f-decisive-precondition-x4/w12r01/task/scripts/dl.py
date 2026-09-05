import os
os.environ.setdefault("HF_HOME","/home/ben/hf_cache")
from huggingface_hub import hf_hub_download
for f in ["main/train-00000-of-00001.parquet"]:
    print(hf_hub_download("openai/gsm8k", f, repo_type="dataset"))
print(hf_hub_download("microsoft/orca-math-word-problems-200k","data/train-00000-of-00001.parquet",repo_type="dataset"))
for i in range(4):
    print(hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset"))
print("DONE")
