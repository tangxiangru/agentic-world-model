#!/usr/bin/env python3
"""Write the frozen batch decode config into a checkpoint directory.

evaluate.py passes no sampling parameters, so vLLM takes its defaults from the
checkpoint's generation_config.json (vllm/config/model.py get_diff_sampling_param
reads temperature/top_k/top_p/min_p/repetition_penalty and ignores do_sample).
The batch protocol is greedy, so every checkpoint that gets scored carries
temperature 0.0 and no top_k/top_p.  transformers' GenerationConfig.validate()
rejects temperature 0.0, so this is written as raw json after training, never
loaded into a Trainer.
"""
import json
import os
import shutil
import sys

SNAP = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
# Trainer(processing_class=tok) writes tokenizer files only; vLLM builds a
# processor for Gemma3ForConditionalGeneration at engine init and needs these.
PROCESSOR_FILES = ["preprocessor_config.json", "processor_config.json"]

GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "transformers_version": "4.50.0.dev0",
}

for d in sys.argv[1:]:
    p = os.path.join(d, "generation_config.json")
    with open(p, "w") as f:
        json.dump(GREEDY, f, indent=2)
    print("greedy decode written to", p)
    for name in PROCESSOR_FILES:
        dst = os.path.join(d, name)
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(SNAP, name), dst)
            print("copied", name, "->", d)
