"""Write the model's recommended decoding defaults (greedy) into generation_config.json.

vLLM reads temperature/top_p/top_k/repetition_penalty from the model's own
generation_config.json when the request does not specify them (which is the case
for the eval harness). The pretrained default is temperature=1.0 sampling; greedy
decoding is the right default for single-sample math accuracy.
"""
import json
import sys

path = sys.argv[1].rstrip("/") + "/generation_config.json"
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

cfg = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    # temperature == 0 makes vLLM decode greedily (a strictly positive value below
    # 0.01 would instead be clamped up to 0.01); do_sample=True keeps
    # transformers' GenerationConfig validator happy on both load and save.
    "do_sample": True,
    "temperature": temp,
    "top_p": 1.0,
    "top_k": 0,
    "transformers_version": "4.57.3",
}
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
print("wrote", path, cfg)

from transformers import GenerationConfig  # noqa: E402
gc = GenerationConfig.from_pretrained(sys.argv[1])
print("reload diff:", gc.to_diff_dict())
