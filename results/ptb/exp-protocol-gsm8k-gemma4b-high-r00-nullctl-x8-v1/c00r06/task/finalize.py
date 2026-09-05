#!/usr/bin/env python3
"""Package a checkpoint dir into a self-contained, vLLM-loadable model dir."""
import argparse, json, os, shutil

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")

GEN_CFG = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "do_sample": False,
    "temperature": 0.0,
    "transformers_version": "4.50.0.dev0",
}

WEIGHTS = ("model.safetensors.index.json", "config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tokenizer-src", default=None,
                    help="dir to take tokenizer files from (default: src, else BASE)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    # model weights + config
    for f in os.listdir(args.src):
        if f.endswith(".safetensors") or f in WEIGHTS:
            shutil.copy(os.path.join(args.src, f), os.path.join(args.dst, f))

    tsrc = args.tokenizer_src or args.src
    tok_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json"]
    for f in tok_files:
        p = os.path.join(tsrc, f)
        if not os.path.exists(p):
            p = os.path.join(BASE, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.dst, f))
    for f in ["preprocessor_config.json", "processor_config.json"]:
        shutil.copy(os.path.join(BASE, f), os.path.join(args.dst, f))
    # chat template embedded for convenience (evaluate.py passes its own)
    shutil.copy("templates/gemma3.jinja", os.path.join(args.dst, "chat_template.jinja"))

    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(GEN_CFG, f, indent=2)

    print("packaged", args.src, "->", args.dst)
    print(sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
