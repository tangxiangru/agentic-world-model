from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

paths = []
for i in range(3):
    p = hf_hub_download(
        "nvidia/OpenMathInstruct-2",
        f"data/train_1M-0000{i}-of-00003.parquet",
        repo_type="dataset",
    )
    print("got", p, flush=True)
    paths.append(p)

t = pq.read_table(paths[0])
print(t.schema, flush=True)
print(t.slice(0, 2).to_pylist(), flush=True)
import collections

print(collections.Counter(t.column("problem_source").to_pylist()), flush=True)
