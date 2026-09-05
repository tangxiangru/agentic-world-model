#!/usr/bin/env python3
"""Create a decode-config variant of a checkpoint: symlink every weight/config
file into a new directory and overwrite generation_config.json.

vLLM (0.11) applies the model's generation_config.json as the server's default
sampling params (ModelConfig.get_diff_sampling_param), and inspect_ai's vllm
provider sends no temperature of its own, so this file decides how the graded
generation is decoded.
"""
import argparse, json, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--gen-config", required=True, help="json string")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.src):
    if f == "generation_config.json":
        continue
    s, d = os.path.join(os.path.abspath(a.src), f), os.path.join(a.dst, f)
    if os.path.exists(d) or os.path.islink(d):
        os.remove(d)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        os.symlink(s, d)

gc = json.load(open(os.path.join(a.src, "generation_config.json")))
gc.update(json.loads(a.gen_config))
for k in list(gc):
    if gc[k] is None:
        del gc[k]
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print(json.dumps(gc, indent=2))
