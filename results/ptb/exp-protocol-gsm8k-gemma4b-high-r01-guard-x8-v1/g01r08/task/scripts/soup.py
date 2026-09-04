"""Uniform weight average of two checkpoints (same architecture)."""
import argparse, json, os, shutil, torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--alpha", type=float, default=0.5, help="weight on A")
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
idx = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
files = sorted(set(idx["weight_map"].values()))
for f in files:
    sa = load_file(os.path.join(args.a, f))
    sb = load_file(os.path.join(args.b, f))
    assert set(sa) == set(sb), f
    out = {}
    for k in sa:
        ta, tb = sa[k].to(torch.float32), sb[k].to(torch.float32)
        out[k] = (args.alpha * ta + (1 - args.alpha) * tb).to(sa[k].dtype)
    save_file(out, os.path.join(args.out, f), metadata={"format": "pt"})
    print("wrote", f, flush=True)
for name in ["config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "tokenizer.model", "preprocessor_config.json", "processor_config.json",
             "generation_config.json"]:
    src = os.path.join(args.a, name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(args.out, name))
print("done", args.out)
