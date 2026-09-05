import os, time
from huggingface_hub import hf_hub_download
t0=time.time()
files=[]
for i in range(4):
    f=hf_hub_download("nvidia/OpenMathInstruct-2", f"data/train-{i:05d}-of-00032.parquet", repo_type="dataset")
    print("got", f, round(time.time()-t0,1), flush=True)
    files.append(f)
for rid, fn in [("openai/gsm8k","main/train-00000-of-00001.parquet"),
                ("microsoft/orca-math-word-problems-200k","data/train-00000-of-00001.parquet")]:
    f=hf_hub_download(rid, fn, repo_type="dataset"); print("got", f, flush=True)
print("done", round(time.time()-t0,1))
