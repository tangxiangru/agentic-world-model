import os
os.environ.setdefault("HF_HOME","/home/ben/hf_cache")
from huggingface_hub import hf_hub_download
for i in range(8,32):
    p=hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset")
    print(i, flush=True)
print("DONE", flush=True)
