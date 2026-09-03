"""Weight-average two checkpoints (uniform model soup) into a new model dir."""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

a_dir, b_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
w = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5   # weight on a_dir

os.makedirs(out_dir, exist_ok=True)
index = json.load(open(os.path.join(a_dir, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))

for shard in shards:
    ta = load_file(os.path.join(a_dir, shard))
    tb = load_file(os.path.join(b_dir, shard))
    assert ta.keys() == tb.keys(), shard
    merged = {}
    for k in ta:
        if ta[k].is_floating_point():
            merged[k] = (ta[k].to(torch.float32) * w
                         + tb[k].to(torch.float32) * (1 - w)).to(ta[k].dtype)
        else:
            merged[k] = ta[k]
    save_file(merged, os.path.join(out_dir, shard), metadata={"format": "pt"})
    print("merged", shard)

for fn in os.listdir(b_dir):
    if fn.endswith(".safetensors") or fn == "training_args.bin":
        continue
    shutil.copy(os.path.join(b_dir, fn), os.path.join(out_dir, fn))
print("wrote", out_dir, sorted(os.listdir(out_dir)))
