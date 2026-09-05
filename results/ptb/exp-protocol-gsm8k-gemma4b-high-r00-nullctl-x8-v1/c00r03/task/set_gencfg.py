#!/usr/bin/env python3
"""Write a deterministic (greedy) generation_config.json into a model dir."""
import json, sys, os

d = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "greedy"
p = os.path.join(d, "generation_config.json")
cfg = json.load(open(p)) if os.path.exists(p) else {}
cfg.update({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
            "cache_implementation": "hybrid"})
if mode == "greedy":
    cfg.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    cfg.pop("min_p", None)
else:
    cfg.update({"do_sample": True, "temperature": float(mode), "top_p": 0.95,
                "top_k": 64})
json.dump(cfg, open(p, "w"), indent=2)
print(json.dumps(cfg))
