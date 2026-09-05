#!/usr/bin/env python3
"""Rewrite a checkpoint's generation_config.json so vLLM defaults to greedy decoding.

The eval harness sends no temperature, so vLLM falls back to the model's
generation_config (the base model ships do_sample=true / top_p=0.95 / top_k=64,
i.e. temperature 1.0 sampling). Greedy is strictly better for math.
"""
import json
import sys

GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "transformers_version": "4.50.0.dev0",
}

if __name__ == "__main__":
    path = sys.argv[1].rstrip("/") + "/generation_config.json"
    with open(path, "w") as f:
        json.dump(GREEDY, f, indent=2)
    print("wrote", path, json.dumps(GREEDY))
