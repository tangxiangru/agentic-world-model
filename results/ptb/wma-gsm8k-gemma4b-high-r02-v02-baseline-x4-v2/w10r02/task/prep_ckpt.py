#!/usr/bin/env python3
"""Make a Trainer checkpoint-N dir loadable by the grader's vllm: copy the
tokenizer/processor files and the greedy generation_config from the run dir."""
import json, os, shutil, sys

SNAPSHOT = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
            "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
GEN = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
       "do_sample": False, "temperature": 0.0, "cache_implementation": "hybrid",
       "transformers_version": "4.50.0.dev0"}

def prep(d, sampled=False):
    for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json",
              "preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAPSHOT, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(d, f)):
            shutil.copy(os.path.realpath(src), os.path.join(d, f))
    gen = dict(GEN)
    if sampled:
        gen = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
               "do_sample": True, "top_k": 64, "top_p": 0.95,
               "cache_implementation": "hybrid", "transformers_version": "4.50.0.dev0"}
    with open(os.path.join(d, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    print("prepped", d, "->", json.dumps(gen))

if __name__ == "__main__":
    prep(sys.argv[1], sampled=(len(sys.argv) > 2 and sys.argv[2] == "sampled"))
