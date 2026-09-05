#!/usr/bin/env python3
"""Copy a trained checkpoint into a deployable model dir (greedy decoding by default)."""
import argparse, json, os, shutil, sys

GEN_CFG = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "transformers_version": "4.57.3",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--greedy", type=int, default=1)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)

    if args.greedy:
        with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
            json.dump(GEN_CFG, f, indent=2)

    # make sure the eval chat template is baked in
    with open("templates/gemma3.jinja") as f:
        ct = f.read()
    p = os.path.join(args.dst, "tokenizer_config.json")
    tc = json.load(open(p))
    tc["chat_template"] = ct
    json.dump(tc, open(p, "w"), indent=2)
    with open(os.path.join(args.dst, "chat_template.jinja"), "w") as f:
        f.write(ct)
    print("wrote", args.dst, sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
