from huggingface_hub import hf_hub_download, list_repo_files
import sys
r='nvidia/OpenMathInstruct-2'
fs=[f for f in list_repo_files(r,repo_type='dataset') if f.endswith('.parquet')]
print(len(fs), fs)
for f in fs[:4]:
    p=hf_hub_download(r,f,repo_type='dataset')
    print('ok',p)
