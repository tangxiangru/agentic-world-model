"""Weighted average of N checkpoints of the same architecture."""
import argparse, json, os, shutil
from safetensors.torch import load_file, save_file

def load_all(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    sd = {}
    for shard in sorted(set(idx["weight_map"].values())):
        sd.update(load_file(os.path.join(path, shard)))
    return sd, idx

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True, help="path:weight")
ap.add_argument("--out", required=True)
a = ap.parse_args()
specs = [(m.rsplit(":", 1)[0], float(m.rsplit(":", 1)[1])) for m in a.models]
tot = sum(w for _, w in specs)
acc, idx = None, None
for p, w in specs:
    sd, i = load_all(p)
    if acc is None:
        acc, idx, ref = {k: v.float() * (w / tot) for k, v in sd.items()}, i, {k: v.dtype for k, v in sd.items()}
    else:
        assert set(sd) == set(acc)
        for k in acc:
            acc[k] += sd[k].float() * (w / tot)
    del sd
os.makedirs(a.out, exist_ok=True)
shards = {}
for k, shard in idx["weight_map"].items():
    shards.setdefault(shard, {})[k] = acc[k].to(ref[k])
for shard, tensors in shards.items():
    save_file(tensors, os.path.join(a.out, shard), metadata={"format": "pt"})
src0 = specs[0][0]
for fn in os.listdir(src0):
    if fn.endswith(".safetensors") or fn == "train_log.jsonl":
        continue
    s = os.path.realpath(os.path.join(src0, fn))
    if os.path.isfile(s):
        shutil.copy(s, os.path.join(a.out, fn))
print("souped", len(acc), "tensors from", specs, "->", a.out)
