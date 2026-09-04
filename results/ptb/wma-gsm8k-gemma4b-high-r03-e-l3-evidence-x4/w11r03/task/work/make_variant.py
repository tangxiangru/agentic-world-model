"""Make a decode variant of a checkpoint: same weights (hard-linked), different
generation_config.json.

vLLM runs with --generation-config auto, so the sampling parameters it applies
come from the checkpoint's own generation_config.json (get_diff_sampling_param
in vllm/config).  evaluate.py sends no temperature, so whatever this file says
is what the grader decodes with.
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--temperature", type=float, default=0.0)
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
        shutil.copy(s, d)

gc = json.load(open(os.path.join(a.src, "generation_config.json")))
gc.pop("top_p", None)
gc.pop("top_k", None)
gc["do_sample"] = a.temperature > 0
gc["temperature"] = a.temperature
if a.top_p is not None:
    gc["top_p"] = a.top_p
if a.top_k is not None:
    gc["top_k"] = a.top_k
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print(json.dumps(gc, indent=2))
assert gc.get("eos_token_id") == [1, 106], f"eos_token_id changed: {gc.get('eos_token_id')}"
print("wrote", a.dst)
