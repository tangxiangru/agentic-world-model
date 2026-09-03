"""Uniform weight average of two checkpoints with identical architecture."""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--weight-a", type=float, default=0.5)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
index = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
for shard in shards:
    ta = load_file(os.path.join(args.a, shard))
    tb = load_file(os.path.join(args.b, shard))
    assert set(ta) == set(tb), shard
    merged = {
        k: (args.weight_a * ta[k].float() + (1 - args.weight_a) * tb[k].float()).to(ta[k].dtype)
        for k in ta
    }
    save_file(merged, os.path.join(args.out, shard), metadata={"format": "pt"})
    print("merged", shard, flush=True)
for f in os.listdir(args.a):
    if f.endswith(".safetensors"):
        continue
    shutil.copy2(os.path.join(args.a, f), os.path.join(args.out, f))
print("wrote", args.out)
