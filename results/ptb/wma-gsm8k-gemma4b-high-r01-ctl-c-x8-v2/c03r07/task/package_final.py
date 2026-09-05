#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ with the greedy decode config, then
prove it loads the way the grader will load it.

Guards the final_model_not_loadable pitfall: real files (never symlinks), the
tokenizer and processor files alongside the weights, an explicit
generation_config.json (temperature 0.0, eos_token_id [1, 106]), and a CPU
load with transformers from a fresh process before anything is declared done.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

GREEDY_GENERATION_CONFIG = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "transformers_version": "4.57.3",
}

SKIP = {"training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)

    for name in sorted(os.listdir(args.src)):
        if name in SKIP or name == "generation_config.json":
            continue
        src = os.path.realpath(os.path.join(args.src, name))
        if not os.path.isfile(src):
            continue
        shutil.copyfile(src, os.path.join(args.dst, name))
        print(f"copied {name} ({os.path.getsize(src)/2**20:.1f} MiB)")

    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(GREEDY_GENERATION_CONFIG, f, indent=2)
    print("wrote greedy generation_config.json")

    # --- verify the way the grader will ---
    import torch
    from transformers import AutoTokenizer, GenerationConfig, Gemma3ForConditionalGeneration

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    assert "gemma" in cfg["architectures"][0].lower(), cfg["architectures"]
    assert cfg.get("dtype", cfg.get("torch_dtype")) == "bfloat16", cfg.get("dtype")

    gen = GenerationConfig.from_pretrained(args.dst).to_diff_dict()
    assert gen.get("temperature") == 0.0, gen
    assert gen.get("eos_token_id") == [1, 106], gen
    assert "top_k" not in gen and "top_p" not in gen, gen
    print("generation config vLLM will read:", gen)

    tok = AutoTokenizer.from_pretrained(args.dst)
    assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.dst, dtype=torch.bfloat16, device_map="cpu"
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded on CPU with transformers: {n/1e9:.2f}B params, dtype {model.dtype}")
    print("OK", os.path.abspath(args.dst))


if __name__ == "__main__":
    main()
