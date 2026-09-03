#!/usr/bin/env python3
"""Create a decoding-config variant of a checkpoint: hard-link the weights,
rewrite only generation_config.json. vLLM reads that file for its default
sampling params (ModelConfig.get_diff_sampling_param), and the grader never
sets temperature/top_p/top_k explicitly, so this file decides how it decodes.
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--top-p", type=float, default=None)
ap.add_argument("--top-k", type=int, default=None)
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.src):
    s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
    if os.path.exists(d):
        os.remove(d)
    if f == "generation_config.json":
        continue
    try:
        os.link(s, d)
    except OSError:
        import shutil
        shutil.copy(s, d)

gc = json.load(open(os.path.join(a.src, "generation_config.json")))
if a.greedy:
    gc.update({"do_sample": False, "temperature": 0.0, "top_k": 1, "top_p": 1.0})
if a.temperature is not None:
    gc["temperature"] = a.temperature
    gc["do_sample"] = a.temperature > 0
if a.top_p is not None:
    gc["top_p"] = a.top_p
if a.top_k is not None:
    gc["top_k"] = a.top_k
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print("wrote", a.dst, gc)
