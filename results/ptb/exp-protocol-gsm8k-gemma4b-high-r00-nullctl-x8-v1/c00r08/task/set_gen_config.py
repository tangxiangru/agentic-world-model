"""Write the decoding defaults that vLLM picks up from a model dir.

inspect_ai sends no temperature/top_p, so `vllm serve` falls back to the model's
generation_config.json. The base gemma-3-4b-pt config asks for sampling
(top_k=64, top_p=0.95, temperature unset -> 1.0); greedy is better for GSM8K.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=-1)
a = ap.parse_args()

cfg = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "do_sample": a.temperature > 0,
    "temperature": a.temperature,
    "top_p": a.top_p,
    "top_k": a.top_k,
    "transformers_version": "4.50.0.dev0",
}
p = os.path.join(a.model_dir, "generation_config.json")
with open(p, "w") as f:
    json.dump(cfg, f, indent=2)
print("wrote", p, cfg)
