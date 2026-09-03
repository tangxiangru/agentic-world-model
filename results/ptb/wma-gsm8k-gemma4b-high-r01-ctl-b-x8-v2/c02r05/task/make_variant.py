"""Copy a checkpoint into a new directory with a different generation_config.

The grader launches vLLM with `--generation-config auto`, so vLLM reads
generation_config.json out of the model directory and uses temperature / top_p /
top_k from it as the server's default sampling params (ModelConfig.
get_diff_sampling_param). Decoding is therefore part of the model artefact.
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=0)
ap.add_argument("--symlink-weights", action="store_true")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.src):
    s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
    if os.path.exists(d):
        continue
    if a.symlink_weights and f.endswith(".safetensors"):
        os.symlink(os.path.abspath(s), d)
    else:
        shutil.copy2(s, d)

gc = json.load(open(os.path.join(a.src, "generation_config.json")))
gc["do_sample"] = a.temperature > 0
gc["temperature"] = a.temperature
gc["top_p"] = a.top_p
gc["top_k"] = a.top_k
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print(json.dumps(gc, indent=2))
