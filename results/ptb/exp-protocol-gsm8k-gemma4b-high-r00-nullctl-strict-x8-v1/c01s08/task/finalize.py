#!/usr/bin/env python3
"""Copy processor configs from the base snapshot into a trained checkpoint and
set a deterministic generation_config so vLLM defaults to greedy decoding."""
import argparse, json, os, shutil

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
EXTRA = ["preprocessor_config.json", "processor_config.json", "added_tokens.json",
         "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
         "tokenizer_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--greedy", type=int, default=1)
    args = ap.parse_args()
    for f in EXTRA:
        src, dst = os.path.join(BASE, f), os.path.join(args.ckpt, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print("copied", f)
    gc_path = os.path.join(args.ckpt, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc.update({"bos_token_id": 2, "pad_token_id": 0, "eos_token_id": [1, 106],
               "cache_implementation": "hybrid"})
    if args.greedy:
        gc.update({"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 1})
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("generation_config:", gc)


if __name__ == "__main__":
    main()
