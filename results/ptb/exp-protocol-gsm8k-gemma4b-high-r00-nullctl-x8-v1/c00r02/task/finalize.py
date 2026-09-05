#!/usr/bin/env python3
"""Copy processor/tokenizer assets from the base snapshot into a trained model dir
and write a greedy generation_config.json."""
import argparse, json, os, shutil

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = [
    "preprocessor_config.json", "processor_config.json", "tokenizer.json",
    "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    for f in NEEDED:
        src = os.path.join(SNAPSHOT, f)
        dst = os.path.join(args.model_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print("copied", f)
    gc = {
        "bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
        "cache_implementation": "hybrid",
    }
    if args.temperature <= 0:
        gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    else:
        gc.update({"do_sample": True, "temperature": args.temperature,
                   "top_p": 0.95, "top_k": 64})
    with open(os.path.join(args.model_dir, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    print("wrote generation_config.json", gc)


if __name__ == "__main__":
    main()
