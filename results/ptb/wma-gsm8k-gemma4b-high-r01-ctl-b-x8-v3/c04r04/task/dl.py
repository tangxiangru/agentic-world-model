from huggingface_hub import hf_hub_download, snapshot_download
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
for i in [0,1,2,3]:
    p=hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset", revision="469216e3f46f4dacf476b382e192485ea51a143e")
    print("ok",p, flush=True)
p=snapshot_download("openai/gsm8k", repo_type="dataset", revision="740312add88f781978c0658806c59bc2815b9866")
print("gsm8k", p, flush=True)
