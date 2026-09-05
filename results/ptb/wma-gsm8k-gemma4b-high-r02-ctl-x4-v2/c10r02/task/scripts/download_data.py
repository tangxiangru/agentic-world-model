import os
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from huggingface_hub import hf_hub_download
files = [f"data/train-{i:05d}-of-00032.parquet" for i in range(6)]
for f in files:
    p = hf_hub_download("nvidia/OpenMathInstruct-2", f, repo_type="dataset")
    print("OK", p, flush=True)
for f in ["main/train-00000-of-00001.parquet"]:
    p = hf_hub_download("openai/gsm8k", f, repo_type="dataset")
    print("OK", p, flush=True)
