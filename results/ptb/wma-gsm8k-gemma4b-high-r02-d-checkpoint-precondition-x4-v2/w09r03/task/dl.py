from huggingface_hub import hf_hub_download
import sys
files = [f"data/train-{i:05d}-of-00032.parquet" for i in range(0,32,1)]
for f in files[:6]:
    p = hf_hub_download("nvidia/OpenMathInstruct-2", f, repo_type="dataset")
    print("ok", p, flush=True)
