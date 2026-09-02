from huggingface_hub import hf_hub_download

for f in [
    "data/train_1M-00002-of-00003.parquet",
    "data/train_2M-00000-of-00006.parquet",
    "data/train_2M-00001-of-00006.parquet",
]:
    p = hf_hub_download("nvidia/OpenMathInstruct-2", f, repo_type="dataset")
    print("OK", p, flush=True)
print("DL_DONE")
