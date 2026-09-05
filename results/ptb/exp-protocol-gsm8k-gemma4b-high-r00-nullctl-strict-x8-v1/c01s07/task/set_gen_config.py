#!/usr/bin/env python3
"""Set the decoding defaults that vLLM picks up from a model's generation_config.json."""
from __future__ import annotations

import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--top-p", type=float, default=None)
ap.add_argument("--top-k", type=int, default=None)
a = ap.parse_args()

p = os.path.join(a.model_dir, "generation_config.json")
cfg = json.load(open(p))
if a.temperature is not None:
    cfg["temperature"] = a.temperature
if a.top_p is not None:
    cfg["top_p"] = a.top_p
if a.top_k is not None:
    cfg["top_k"] = a.top_k
json.dump(cfg, open(p, "w"), indent=2)
print(json.dumps(cfg))
