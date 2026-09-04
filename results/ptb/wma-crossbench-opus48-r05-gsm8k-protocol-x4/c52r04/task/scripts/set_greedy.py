#!/usr/bin/env python3
"""Set a checkpoint's generation_config.json to greedy decoding (do_sample=false)."""
import json, sys

path = sys.argv[1]  # path to generation_config.json
cfg = json.load(open(path))
cfg["do_sample"] = False
cfg["temperature"] = 0.0
cfg.pop("top_p", None)
cfg.pop("top_k", None)
json.dump(cfg, open(path, "w"), indent=2)
print(f"[greedy] {path}: {cfg}")
