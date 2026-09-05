from huggingface_hub import snapshot_download
p1 = snapshot_download("nvidia/OpenMathInstruct-2", repo_type="dataset",
                       allow_patterns=["data/train_1M-*.parquet","README.md"],
                       revision="469216e3f46f4dacf476b382e192485ea51a143e")
print("OMI2", p1, flush=True)
p2 = snapshot_download("microsoft/orca-math-word-problems-200k", repo_type="dataset",
                       allow_patterns=["data/*.parquet","README.md"],
                       revision="29255d1770cc4eac66e5e7fa378cba542c026350")
print("ORCA", p2, flush=True)
p3 = snapshot_download("openai/gsm8k", repo_type="dataset")
print("GSM8K", p3, flush=True)
