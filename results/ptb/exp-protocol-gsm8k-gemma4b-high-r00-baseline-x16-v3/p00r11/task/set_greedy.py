#!/usr/bin/env python3
"""Declare greedy decoding in a checkpoint's generation_config.json.

The grading harness sends no sampling parameters, so vLLM falls back to the
defaults it reads from this file (temperature / top_k / top_p; `do_sample` is
ignored by vLLM).  temperature 0.0 with top_k 1 makes vLLM decode greedily and
still passes transformers' GenerationConfig validation, which refuses
do_sample=False together with temperature/top_k.
"""
from __future__ import annotations

import json
import sys

for path in sys.argv[1:]:
    f = f"{path}/generation_config.json"
    g = json.load(open(f))
    g.update({"do_sample": True, "temperature": 0.0, "top_k": 1, "top_p": 1.0})
    json.dump(g, open(f, "w"), indent=2)
    print("greedy:", f)
