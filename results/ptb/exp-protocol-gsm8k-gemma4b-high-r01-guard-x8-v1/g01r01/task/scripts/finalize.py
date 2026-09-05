"""Assemble final_model/ from a checkpoint: real files (no symlinks), tokenizer,
greedy generation_config, then a CPU load check with transformers."""
import argparse, json, os, shutil, sys
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", default="/home/ben/task/final_model")
a = ap.parse_args()

GREEDY = {"bos_token_id": 2, "cache_implementation": "hybrid", "do_sample": False,
          "eos_token_id": [1, 106], "pad_token_id": 0, "temperature": 0.0,
          "top_k": -1, "top_p": 1.0, "transformers_version": "4.57.3"}

if os.path.exists(a.dst):
    shutil.rmtree(a.dst)
os.makedirs(a.dst)
for f in sorted(os.listdir(a.src)):
    if f in ("training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth",
             "trainer_state.json", "generation_config.json"):
        continue
    src = os.path.realpath(os.path.join(a.src, f))
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(a.dst, f))
SNAP = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
for extra in ("preprocessor_config.json", "processor_config.json", "tokenizer.model"):
    p = os.path.join(a.dst, extra)
    if not os.path.exists(p) and os.path.exists(os.path.join(SNAP, extra)):
        shutil.copy(os.path.join(SNAP, extra), p)
with open(os.path.join(a.dst, "generation_config.json"), "w") as f:
    json.dump(GREEDY, f, indent=2)
print("files:", sorted(os.listdir(a.dst)))

# load check on CPU, exactly as a fresh process would
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
cfg = AutoConfig.from_pretrained(a.dst)
print("architectures:", cfg.architectures)
tok = AutoTokenizer.from_pretrained(a.dst)
m = AutoModelForCausalLM.from_pretrained(a.dst, dtype=torch.bfloat16)
n = sum(p.numel() for p in m.parameters())
print(f"loaded ok on CPU: {type(m).__name__} {n/1e9:.2f}B params")
print("generation_config:", json.load(open(os.path.join(a.dst, "generation_config.json"))))
