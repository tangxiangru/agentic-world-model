#!/usr/bin/env python3
"""Copy a checkpoint to final_model/ and verify it the way the grader will use it.

Also writes the generation_config the grader's vLLM will pick up as its default
sampling params ("Default sampling parameters have been overridden by the
model's Hugging Face generation config" in the eval log).
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--decode", choices=["greedy", "base"], default="greedy")
    ap.add_argument("--verify", type=int, default=1)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)
    for junk in ["training_args.bin", "train_metrics.json"]:
        p = os.path.join(args.dst, junk)
        if os.path.exists(p):
            os.remove(p)

    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
    }
    if args.decode == "greedy":
        gc["do_sample"] = False
        gc["temperature"] = 0.0
    else:
        gc.update({"do_sample": True, "top_k": 64, "top_p": 0.95})
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    print("wrote generation_config:", gc, flush=True)

    if args.verify:
        import torch
        from transformers import AutoConfig, AutoTokenizer

        cfg = AutoConfig.from_pretrained(args.dst)
        print("architectures:", cfg.architectures, flush=True)
        tok = AutoTokenizer.from_pretrained(args.dst)
        print("tokenizer ok, eos:", tok.eos_token, "end_of_turn id:",
              tok.convert_tokens_to_ids("<end_of_turn>"), flush=True)
        from safetensors import safe_open

        idx = json.load(open(os.path.join(args.dst, "model.safetensors.index.json")))
        files = sorted(set(idx["weight_map"].values()))
        n = 0
        for fn in files:
            with safe_open(os.path.join(args.dst, fn), framework="pt") as f:
                for k in f.keys():
                    n += 1
        print(f"{len(files)} shards, {n} tensors present", flush=True)
        print("files:", sorted(os.listdir(args.dst)), flush=True)


if __name__ == "__main__":
    main()
