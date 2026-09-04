#!/usr/bin/env python3
"""Make a mid-run Trainer checkpoint loadable and greedily-decoded.

Trainer's intermediate checkpoints carry neither the tokenizer/processor files
nor the greedy generation_config that vLLM reads, so they cannot be scored
as-is. Copy both in from a sibling `final/` directory.
"""
import json, os, shutil, sys

ckpt, src = sys.argv[1], sys.argv[2]
for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json",
          "preprocessor_config.json", "processor_config.json"]:
    p = os.path.join(src, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(ckpt, f))
json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
           "cache_implementation": "hybrid", "do_sample": False,
           "temperature": 0.0, "transformers_version": "4.57.3"},
          open(os.path.join(ckpt, "generation_config.json"), "w"), indent=2)
print("prepared", ckpt)
