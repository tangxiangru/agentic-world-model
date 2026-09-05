#!/usr/bin/env python3
"""Write a deterministic (greedy) generation_config.json into a model dir.

vLLM reads generation_config.json and uses temperature/top_p/top_k/min_p as the
default sampling params for requests that don't specify them, so this is what
selects greedy decoding for the served model.
"""
import json, sys

path = sys.argv[1]
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
cfg = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": temp > 0,
    "temperature": temp,
    "top_p": 1.0,
    "top_k": -1,
    "transformers_version": "4.57.3",
}
with open(f"{path}/generation_config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("wrote", f"{path}/generation_config.json", cfg)
