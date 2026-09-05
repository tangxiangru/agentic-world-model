#!/usr/bin/env python3
"""Build a gradeable model directory: weights from a checkpoint, greedy decode.

Why this exists. evaluate.py sends no temperature/top_p/top_k, so vLLM takes its
default sampling params from the model directory's own generation_config.json
(ModelConfig.get_diff_sampling_param -> ChatCompletionRequest.to_sampling_params).
train_sft.py copies the BASE file there, which says do_sample true / top_k 64 /
top_p 0.95, i.e. temperature-1.0 sampling. exp-03 measured that swap at +8 points.

Trainer's intermediate checkpoint-N/ dirs hold weights only and are fp32, so this
also brings the tokenizer/processor files across and can cast to bf16.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "transformers_version": "4.50.0.dev0",
}

AUX = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
       "special_tokens_map.json", "added_tokens.json",
       "preprocessor_config.json", "processor_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkpoint dir with weights + config.json")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--copy", action="store_true", help="real copies instead of symlinks")
    ap.add_argument("--cast-bf16", action="store_true", help="re-save weights in bf16 (fp32 Trainer checkpoints)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    if args.cast_bf16:
        import torch
        from transformers import Gemma3ForConditionalGeneration
        m = Gemma3ForConditionalGeneration.from_pretrained(args.src, dtype=torch.bfloat16)
        m.config.use_cache = True
        m.save_pretrained(args.dst, safe_serialization=True)
    else:
        for f in glob.glob(os.path.join(args.src, "*")):
            b = os.path.basename(f)
            if b in ("generation_config.json",) or os.path.isdir(f):
                continue
            d = os.path.join(args.dst, b)
            if os.path.lexists(d):
                os.remove(d)
            (shutil.copy if args.copy else os.symlink)(os.path.realpath(f), d)

    for fn in AUX:
        d = os.path.join(args.dst, fn)
        if os.path.exists(d):
            continue
        for src_dir in (args.src, BASE):
            s = os.path.join(src_dir, fn)
            if os.path.exists(s):
                shutil.copy(os.path.realpath(s), d)
                break

    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(GREEDY, f, indent=2)

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    print(f"{args.dst}: architectures={cfg['architectures']} "
          f"files={len(os.listdir(args.dst))} greedy generation_config written")


if __name__ == "__main__":
    main()
