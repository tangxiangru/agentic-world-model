#!/usr/bin/env python3
"""Write an explicit greedy generation_config into a checkpoint dir.

vLLM 0.11's get_diff_sampling_param keeps a field only when it is not None, so
temperature/top_p/top_k must carry real values or vLLM serves its own defaults
(temperature 1.0 sampling). Also pins eos_token_id to [1, 106] so <end_of_turn>
still stops generation after a Trainer save.
"""
import json, os, shutil, sys

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

for dest in sys.argv[1:]:
    p = os.path.join(dest, "generation_config.json")
    gc = json.load(open(p)) if os.path.exists(p) else {}
    gc.update({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
               "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
               "cache_implementation": "hybrid"})
    json.dump(gc, open(p, "w"), indent=2)
    for f in ["preprocessor_config.json", "processor_config.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
              "tokenizer.model"]:
        src, dst = os.path.join(SNAPSHOT, f), os.path.join(dest, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print(dest, json.load(open(p)))
