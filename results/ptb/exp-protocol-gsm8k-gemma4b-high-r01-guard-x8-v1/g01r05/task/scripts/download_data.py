from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="nvidia/OpenMathInstruct-2", repo_type="dataset",
                      allow_patterns=["data/train_1M-*.parquet","README.md"],
                      revision="469216e3f46f4dacf476b382e192485ea51a143e",
                      max_workers=8)
print("OMI2", p)
p2 = snapshot_download(repo_id="openai/gsm8k", repo_type="dataset",
                       revision="740312add88f781978c0658806c59bc2815b9866", max_workers=4)
print("GSM8K", p2)
p3 = snapshot_download(repo_id="meta-math/MetaMathQA", repo_type="dataset",
                       revision="aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18", max_workers=4)
print("MMQA", p3)
print("DONE")
