import os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
REV = "469216e3f46f4dacf476b382e192485ea51a143e"
shards = [f"data/train-{i:05d}-of-00032.parquet" for i in range(int(sys.argv[1]))]
def get(f):
    p = hf_hub_download("nvidia/OpenMathInstruct-2", f, repo_type="dataset", revision=REV)
    print("done", f, flush=True); return p
with ThreadPoolExecutor(8) as ex:
    list(ex.map(get, shards))
print("ALLDONE")
