#!/usr/bin/env python3
"""Turn a Trainer checkpoint into a directory vLLM can serve.

save_only_model=True writes weights + config only; the grader needs the tokenizer,
the Gemma3 processor files, and a generation_config whose temperature makes the
decode greedy (vLLM reads temperature/top_k/top_p out of it -- do_sample is ignored).
"""
import argparse, json, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.ckpt):
    if f.endswith((".safetensors", ".json")) and not f.startswith(("optimizer", "trainer_state", "rng_state")):
        shutil.copyfile(os.path.join(a.ckpt, f), os.path.join(a.out, f))
for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
          "added_tokens.json", "preprocessor_config.json", "processor_config.json"):
    src = os.path.join(a.base, f)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(a.out, f))
gc = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
      "cache_implementation": "hybrid", "do_sample": a.temperature > 0}
if a.temperature is not None:
    gc["temperature"] = a.temperature
json.dump(gc, open(os.path.join(a.out, "generation_config.json"), "w"), indent=2)
print("packaged", a.out, sorted(os.listdir(a.out)))
