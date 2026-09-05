#!/usr/bin/env python3
"""Write the decoding defaults that ship with the model.

vLLM picks its default sampling parameters up from `generation_config.json`.
The base checkpoint ships Gemma's chat defaults (`do_sample`, top_k=64,
top_p=0.95), i.e. temperature-1 sampling, which is the wrong default for a
math-solving model.  Greedy decoding is the right default here.
"""
import json
import sys

GREEDY = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "transformers_version": "4.57.3",
}

SAMPLING = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": True,
    "top_k": 64,
    "top_p": 0.95,
    "transformers_version": "4.57.3",
}

if __name__ == "__main__":
    path, mode = sys.argv[1], sys.argv[2]
    cfg = GREEDY if mode == "greedy" else SAMPLING
    with open(f"{path}/generation_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"{path}/generation_config.json <- {mode}")
