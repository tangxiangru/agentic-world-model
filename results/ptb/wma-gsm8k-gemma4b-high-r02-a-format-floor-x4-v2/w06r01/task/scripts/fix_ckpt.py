#!/usr/bin/env python3
"""Make a Trainer-written checkpoint-N/ directory evaluable.

Trainer re-emits the parent's generation_config (do_sample true, top_k 64,
top_p 0.95) and does not save the tokenizer, so an intermediate checkpoint
would be graded under a different decode than final/.  Copy the tokenizer and
processor files across and write the same greedy generation_config.
"""
import json, os, shutil, sys

SNAP = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
d = sys.argv[1]
for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json",
          "preprocessor_config.json", "processor_config.json"):
    src = os.path.join(SNAP, f)
    if os.path.exists(src) and not os.path.exists(os.path.join(d, f)):
        shutil.copy(src, os.path.join(d, f))
json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
           "cache_implementation": "hybrid",
           "do_sample": False, "temperature": 0.0},
          open(os.path.join(d, "generation_config.json"), "w"), indent=2)
print("fixed", d, sorted(os.listdir(d)))
