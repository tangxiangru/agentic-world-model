#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ as real files and prove it loads."""
import argparse, json, os, shutil, sys

KEEP_SUFFIX = (".json", ".model", ".safetensors")
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = ["config.json", "generation_config.json", "model.safetensors.index.json",
          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "preprocessor_config.json", "processor_config.json", "tokenizer.model"]

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", default="/home/ben/task/final_model")
a = ap.parse_args()

if os.path.exists(a.dst):
    shutil.rmtree(a.dst)
os.makedirs(a.dst)
for f in sorted(os.listdir(a.src)):
    s = os.path.join(a.src, f)
    if os.path.isdir(s) or not f.endswith(KEEP_SUFFIX):
        continue
    shutil.copy(os.path.realpath(s), os.path.join(a.dst, f))
for f in NEEDED:
    d = os.path.join(a.dst, f)
    if not os.path.exists(d) and os.path.exists(os.path.join(SNAP, f)):
        shutil.copy(os.path.join(SNAP, f), d)

missing = [f for f in NEEDED if not os.path.exists(os.path.join(a.dst, f))]
print("files:", sorted(os.listdir(a.dst)))
print("missing:", missing)
print("generation_config:", json.load(open(os.path.join(a.dst, "generation_config.json"))))
cfg = json.load(open(os.path.join(a.dst, "config.json")))
print("architectures:", cfg["architectures"])
assert not missing, missing
assert cfg["architectures"][0].lower().startswith("gemma3")
