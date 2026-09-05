"""Uniform weight average of several checkpoints (model soup)."""
import argparse, json, os, shutil, torch
from safetensors.torch import load_file, save_file
ap = argparse.ArgumentParser()
ap.add_argument("--srcs", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
idx = json.load(open(os.path.join(a.srcs[0], "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
for shard in shards:
    acc = None
    for s in a.srcs:
        sd = load_file(os.path.join(s, shard))
        if acc is None:
            acc = {k: v.to(torch.float32) for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].to(torch.float32)
    acc = {k: (v / len(a.srcs)).to(torch.bfloat16) for k, v in acc.items()}
    save_file(acc, os.path.join(a.out, shard), metadata={"format": "pt"})
    print("wrote", shard, flush=True)
for f in os.listdir(a.srcs[0]):
    if f.endswith(".safetensors") or f in ("training_args.bin", "generation_config.json"):
        continue
    src = os.path.realpath(os.path.join(a.srcs[0], f))
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(a.out, f))
shutil.copy("/home/ben/task/final_model/generation_config.json",
            os.path.join(a.out, "generation_config.json"))
print("soup at", a.out, sorted(os.listdir(a.out)))
