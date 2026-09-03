#!/usr/bin/env python3
"""Copy a chosen checkpoint into final_model/ and verify it the way the grader will load it.

Checks, in order: every file the grader needs is present; config.json's architecture
resolves to 'gemma' through evaluate.py's own model_type(); generation_config asks for
greedy by name and keeps eos_token_id [1, 106]; the weights load with transformers on CPU;
the tokenizer round-trips '<end_of_turn>' to token 106.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = ["config.json", "generation_config.json", "model.safetensors.index.json",
          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
FROM_BASE = ["preprocessor_config.json", "processor_config.json", "added_tokens.json",
             "tokenizer.model"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    if not a.verify_only:
        if os.path.exists(a.dst):
            shutil.rmtree(a.dst)
        os.makedirs(a.dst)
        for fn in os.listdir(a.src):
            if fn in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                      "training_args.bin"):
                continue
            shutil.copy2(os.path.join(a.src, fn), os.path.join(a.dst, fn))
        for fn in FROM_BASE:
            if not os.path.exists(os.path.join(a.dst, fn)) and os.path.exists(os.path.join(BASE, fn)):
                shutil.copy2(os.path.join(BASE, fn), os.path.join(a.dst, fn))
        gc = json.load(open(os.path.join(BASE, "generation_config.json")))
        for k in ("do_sample", "top_k", "top_p"):
            gc.pop(k, None)
        gc["temperature"] = 0.0
        json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
        print(f"copied {a.src} -> {a.dst}")

    fail = []
    for fn in NEEDED:
        if not os.path.exists(os.path.join(a.dst, fn)):
            fail.append(f"missing {fn}")

    cfg = json.load(open(os.path.join(a.dst, "config.json")))
    arch = cfg["architectures"][0].lower()
    if "gemma" not in arch:
        fail.append(f"evaluate.py model_type() would not resolve 'gemma' from {arch}")
    print("architecture:", cfg["architectures"][0])

    gc = json.load(open(os.path.join(a.dst, "generation_config.json")))
    print("generation_config:", gc)
    if gc.get("temperature") != 0.0:
        fail.append("generation_config.temperature is not 0.0 (vLLM would sample)")
    if "do_sample" in gc or "top_k" in gc or "top_p" in gc:
        fail.append("generation_config still carries do_sample/top_k/top_p")
    if gc.get("eos_token_id") != [1, 106]:
        fail.append(f"eos_token_id is {gc.get('eos_token_id')}, expected [1, 106]")

    idx = json.load(open(os.path.join(a.dst, "model.safetensors.index.json")))
    for shard in set(idx["weight_map"].values()):
        if not os.path.exists(os.path.join(a.dst, shard)):
            fail.append(f"missing shard {shard}")

    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.dst)
    ids = tok("ANSWER: 7<end_of_turn>", add_special_tokens=False)["input_ids"]
    print("tokenizer tail ids:", ids[-3:])
    if ids[-1] != 106:
        fail.append("tokenizer does not map '<end_of_turn>' to 106")

    import torch
    m = Gemma3ForConditionalGeneration.from_pretrained(a.dst, torch_dtype=torch.bfloat16,
                                                       device_map="cpu")
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded on CPU: {n/1e9:.2f}B params")
    if n < 4e9:
        fail.append(f"only {n} parameters loaded")

    if fail:
        print("\nFAILED:")
        for f in fail:
            print("  -", f)
        sys.exit(1)
    print("\nfinal_model verification passed")


if __name__ == "__main__":
    main()
