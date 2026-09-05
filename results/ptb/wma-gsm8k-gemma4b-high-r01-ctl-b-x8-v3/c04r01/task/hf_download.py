import os
os.environ.setdefault('HF_HOME','/home/ben/hf_cache')
from huggingface_hub import hf_hub_download
import sys
which=sys.argv[1]
if which=='gsm8k':
    for f in ["main/train-00000-of-00001.parquet"]:
        print(hf_hub_download("openai/gsm8k", f, repo_type="dataset"))
elif which=='meta':
    print(hf_hub_download("meta-math/MetaMathQA","MetaMathQA-395K.json",repo_type="dataset"))
else:
    i=int(which)
    print(hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset"))
