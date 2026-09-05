#!/usr/bin/env python3
"""Set a model dir's generation_config.json to greedy decoding for GSM8K eval."""
import json, sys, os
d = sys.argv[1]
p = os.path.join(d, "generation_config.json")
cfg = json.load(open(p))
cfg["do_sample"] = False
cfg.pop("top_p", None)
cfg.pop("top_k", None)
cfg["temperature"] = 0.0
# keep both eos ids so <end_of_turn> (106) stops generation
cfg["eos_token_id"] = [1, 106]
cfg["pad_token_id"] = 0
cfg["bos_token_id"] = 2
json.dump(cfg, open(p, "w"), indent=2)
print("updated", p, "->", cfg)
