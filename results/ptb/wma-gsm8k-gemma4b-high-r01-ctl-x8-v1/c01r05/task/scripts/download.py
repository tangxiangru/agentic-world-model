from huggingface_hub import hf_hub_download, snapshot_download
import os
os.environ.setdefault("HF_HUB_ENABLE_HXET","1")
# gsm8k train (allowed) - for few-shot reproduction + RFT question pool
p = snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/*"])
print("gsm8k", p, flush=True)
for i in range(8):
    f = hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset")
    print("omi2", i, f, flush=True)
f = hf_hub_download("meta-math/MetaMathQA", "MetaMathQA-395K.json", repo_type="dataset")
print("metamath", f, flush=True)
