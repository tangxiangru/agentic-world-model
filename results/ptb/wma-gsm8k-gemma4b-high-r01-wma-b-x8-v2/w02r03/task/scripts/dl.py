from huggingface_hub import snapshot_download
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","1")
p=snapshot_download("nvidia/OpenMathInstruct-2", repo_type="dataset", allow_patterns=["data/*.parquet"], max_workers=16)
print("OMI2", p)
p2=snapshot_download("openai/gsm8k", repo_type="dataset")
print("GSM8K", p2)
p3=snapshot_download("meta-math/MetaMathQA", repo_type="dataset")
print("MetaMath", p3)
