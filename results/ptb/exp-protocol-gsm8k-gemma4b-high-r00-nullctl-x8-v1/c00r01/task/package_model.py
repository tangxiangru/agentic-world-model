#!/usr/bin/env python3
"""Copy a trained checkpoint into a target dir and set deterministic decoding defaults."""
import argparse, json, os, shutil, sys

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--greedy", action="store_true", default=True)
ap.add_argument("--sampling", dest="greedy", action="store_false")
a = ap.parse_args()

if os.path.exists(a.dst):
    shutil.rmtree(a.dst)
shutil.copytree(a.src, a.dst, ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt",
                                                            "rng_state*", "trainer_state.json",
                                                            "training_args.bin"))
for fn in ["preprocessor_config.json", "processor_config.json", "tokenizer.model"]:
    s = os.path.join(BASE, fn)
    d = os.path.join(a.dst, fn)
    if os.path.exists(s) and not os.path.exists(d):
        shutil.copy(s, d)

gc = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
      "cache_implementation": "hybrid", "transformers_version": "4.50.0.dev0"}
if a.greedy:
    gc.update({"do_sample": False, "temperature": 0.0})
else:
    gc.update({"do_sample": True, "top_k": 64, "top_p": 0.95})
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print("packaged ->", a.dst)
print(sorted(os.listdir(a.dst)))
