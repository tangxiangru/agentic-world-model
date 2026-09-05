#!/usr/bin/env python3
"""Post-process a saved checkpoint so vLLM loads it as bf16.

save_pretrained keeps writing dtype: float32 into config.json even though the
state dict handed to it is bf16, and vLLM's dtype="auto" reads that field and
upcasts. This only touches config.json; the weights are untouched.
"""
import json
import os
import sys

for path in sys.argv[1:]:
    cf = os.path.join(path, "config.json")
    c = json.load(open(cf))
    c["dtype"] = "bfloat16"
    c["torch_dtype"] = "bfloat16"
    if "text_config" in c and isinstance(c["text_config"], dict):
        c["text_config"]["dtype"] = "bfloat16"
        c["text_config"]["torch_dtype"] = "bfloat16"
    if "vision_config" in c and isinstance(c["vision_config"], dict):
        c["vision_config"]["dtype"] = "bfloat16"
        c["vision_config"]["torch_dtype"] = "bfloat16"
    json.dump(c, open(cf, "w"), indent=2)
    print(f"[finalize] {path}: dtype -> bfloat16")
