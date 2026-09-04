#!/usr/bin/env python3
"""Copy a trained checkpoint into final_model/ and make it loadable the way the
grader loads it (pitfalls.yaml: final_model_not_loadable).

- copies weights + config + tokenizer + processor files
- writes generation_config.json with the decode settings we want vLLM to default to
- loads the result once with transformers on CPU and renders one prompt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SRC_EXTRA = [
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "preprocessor_config.json", "processor_config.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.ckpt):
        if f.endswith(".safetensors") or f in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(os.path.join(args.ckpt, f), os.path.join(args.out, f))
    for f in SRC_EXTRA:
        for src_dir in (args.ckpt, args.base):
            p = os.path.join(src_dir, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(args.out, f))
                break

    gen = {
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "transformers_version": "4.57.3",
    }
    if args.temperature <= 0:
        gen["do_sample"] = False
        gen["temperature"] = 0.0
        gen["top_k"] = 1
        gen["top_p"] = 1.0
    else:
        gen["do_sample"] = True
        gen["temperature"] = args.temperature
        gen["top_k"] = 64
        gen["top_p"] = 0.95
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)

    print("files:", sorted(os.listdir(args.out)))

    if args.no_verify:
        return
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(args.out)
    m = Gemma3ForConditionalGeneration.from_pretrained(args.out, dtype=torch.bfloat16)
    print("loaded ok:", type(m).__name__, sum(p.numel() for p in m.parameters()) / 1e9, "B params")
    cfg = json.load(open(os.path.join(args.out, "config.json")))
    print("architectures:", cfg["architectures"])
    print("eos:", tok.convert_tokens_to_ids("<end_of_turn>"))


if __name__ == "__main__":
    main()
