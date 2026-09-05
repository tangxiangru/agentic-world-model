"""Uniform weight average (model soup) of checkpoints on one fine-tuning tree.

Accumulates in fp32 on CPU, writes bf16, and copies the config/tokenizer from the
first source so the result loads exactly like its members.
"""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--tokenizer-from", default="/home/ben/task/ckpts/exp-02/final")
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()


def shards(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        return sorted({v for v in json.load(open(idx))["weight_map"].values()})
    return ["model.safetensors"]


acc, n = {}, len(a.src)
for i, d in enumerate(a.src):
    for sh in shards(d):
        sd = load_file(os.path.join(d, sh))
        for k, v in sd.items():
            f = v.float()
            acc[k] = f if k not in acc else acc[k] + f
        del sd
    print(f"[{i+1}/{n}] accumulated {d}", flush=True)

for k in acc:
    acc[k] = (acc[k] / n).to(torch.bfloat16)
print(f"averaged {len(acc)} tensors over {n} checkpoints", flush=True)

os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.dst):
    p = os.path.join(a.dst, f)
    os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)

# rebuild the same shard layout as the first source
first = a.src[0]
idx_path = os.path.join(first, "model.safetensors.index.json")
if os.path.exists(idx_path):
    idx = json.load(open(idx_path))
    by_shard = {}
    for k, sh in idx["weight_map"].items():
        by_shard.setdefault(sh, {})[k] = acc[k]
    for sh, sd in by_shard.items():
        save_file(sd, os.path.join(a.dst, sh), metadata={"format": "pt"})
    json.dump(idx, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"))
else:
    save_file(acc, os.path.join(a.dst, "model.safetensors"), metadata={"format": "pt"})

shutil.copy(os.path.join(first, "config.json"), os.path.join(a.dst, "config.json"))
for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json"]:
    s = os.path.join(a.tokenizer_from, f)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(a.dst, f))

gc = json.load(open(os.path.join(a.tokenizer_from, "generation_config.json")))
gc.pop("top_p", None)
gc.pop("top_k", None)
gc["do_sample"] = a.temperature > 0
gc["temperature"] = a.temperature
assert gc["eos_token_id"] == [1, 106], gc["eos_token_id"]
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print("wrote", a.dst, sorted(os.listdir(a.dst)))
