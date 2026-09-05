from huggingface_hub import snapshot_download
import sys
print(snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/*"]), flush=True)
print(snapshot_download("microsoft/orca-math-word-problems-200k", repo_type="dataset"), flush=True)
print(snapshot_download("nvidia/OpenMathInstruct-2", repo_type="dataset", allow_patterns=["data/train-0000[0-3]-of-00032.parquet"]), flush=True)
print(snapshot_download("meta-math/MetaMathQA", repo_type="dataset"), flush=True)
print("DONE", flush=True)
