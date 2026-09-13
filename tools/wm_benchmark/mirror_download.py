import os, time, sys
from huggingface_hub import snapshot_download
REV="07132f15e3c6cc6714ae84835b1896d734c5d54a"
tok=os.environ["MY_HF_TOKEN"]; repo="JerrrrryL/awm-gsm8k-trajectories"
t=time.time()
for attempt in range(1,21):
    try:
        p=snapshot_download(repo,repo_type="dataset",revision=REV,token=tok,local_dir="hf-mirror",max_workers=8)
        print(f"DONE attempt {attempt} in {time.time()-t:.0f}s -> {p}", flush=True); break
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True); time.sleep(30)
