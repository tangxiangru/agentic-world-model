"""Hard-link a checkpoint into a sibling dir and give it a different decoder.

vLLM reads temperature / top_p / top_k / min_p / repetition_penalty /
max_new_tokens out of generation_config.json (ModelConfig.get_diff_sampling_param),
so the decoder a model is graded under is a property of its directory. Hard links
mean the 8 GB of weights are not duplicated; generation_config.json is unlinked and
rewritten so the original is untouched.
"""
import argparse, json, os, shutil, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=-1)
a = ap.parse_args()

if os.path.exists(a.dst):
    shutil.rmtree(a.dst)
subprocess.run(["cp", "-al", a.src, a.dst], check=True)
import glob
for p in glob.glob(os.path.join(a.dst, "checkpoint-*")):
    if os.path.isdir(p):
        shutil.rmtree(p)
gcp = os.path.join(a.dst, "generation_config.json")
gc = json.load(open(gcp))
os.remove(gcp)  # break the hard link before writing
gc.update({"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k})
gc.pop("do_sample", None)
assert gc["eos_token_id"] == [1, 106]
json.dump(gc, open(gcp, "w"), indent=2)
print(a.dst, json.dumps(gc))
