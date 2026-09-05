from huggingface_hub import snapshot_download
print(snapshot_download("nvidia/OpenMathInstruct-2", repo_type="dataset",
      allow_patterns=[f"data/train-000{i:02d}-of-00032.parquet" for i in range(4,20)]), flush=True)
print("DONE", flush=True)
