import os, sys
from huggingface_hub import hf_hub_download, snapshot_download
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
what = sys.argv[1]
if what == "gsm8k":
    for f in ["main/train-00000-of-00001.parquet"]:
        print(hf_hub_download("openai/gsm8k", f, repo_type="dataset"))
elif what == "metamath":
    print(hf_hub_download("meta-math/MetaMathQA","MetaMathQA-395K.json", repo_type="dataset"))
elif what == "orca":
    print(hf_hub_download("microsoft/orca-math-word-problems-200k","data/train-00000-of-00001.parquet", repo_type="dataset"))
elif what == "omi2":
    for i in range(int(sys.argv[2])):
        print(hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset"))
