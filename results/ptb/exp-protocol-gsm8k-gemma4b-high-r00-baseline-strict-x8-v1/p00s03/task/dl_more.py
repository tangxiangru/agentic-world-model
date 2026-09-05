from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor
def g(i):
    return hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset")
with ThreadPoolExecutor(8) as ex:
    for r in ex.map(g, range(4,32)): print(r, flush=True)
print("done")
