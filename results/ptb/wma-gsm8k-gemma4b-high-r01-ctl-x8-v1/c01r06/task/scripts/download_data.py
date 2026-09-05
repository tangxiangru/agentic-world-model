import os
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from huggingface_hub import hf_hub_download, snapshot_download

# gsm8k train (allowed: train split only)
for f in ["main/train-00000-of-00001.parquet"]:
    p = hf_hub_download("openai/gsm8k", f, repo_type="dataset")
    print("gsm8k:", p, flush=True)

# OpenMathInstruct-2 shards
for i in range(8):
    f = f"data/train-{i:05d}-of-00032.parquet"
    p = hf_hub_download("nvidia/OpenMathInstruct-2", f, repo_type="dataset")
    print("omi2:", p, flush=True)

p = hf_hub_download("meta-math/MetaMathQA", "MetaMathQA-395K.json", repo_type="dataset")
print("metamath:", p, flush=True)
print("DONE", flush=True)
