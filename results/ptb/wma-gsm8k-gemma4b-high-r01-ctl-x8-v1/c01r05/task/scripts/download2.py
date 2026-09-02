from huggingface_hub import hf_hub_download
for i in range(8, 24):
    f = hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset")
    print("omi2", i, flush=True)
