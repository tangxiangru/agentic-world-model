#!/usr/bin/env python3
"""Assemble final_model/ from a checkpoint and verify it the way the grader will.

Copies (not symlinks - the grader loads final_model/ from a fresh process and a
dangling link would be fatal) the weights, tokenizer and configs, writes the
chosen generation_config.json, then loads the directory once with transformers
on CPU and renders one prompt through templates/gemma3.jinja.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fmt import STOP_TOKEN, load_tokenizer, render_prompt  # noqa: E402

REQUIRED = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        if os.path.exists(args.dst):
            shutil.rmtree(args.dst)
        os.makedirs(args.dst)
        for fn in sorted(os.listdir(args.src)):
            src = os.path.realpath(os.path.join(args.src, fn))
            if os.path.isdir(src):
                continue
            shutil.copy2(src, os.path.join(args.dst, fn))
            print("copied", fn, os.path.getsize(src))

        gc = json.load(open(os.path.join(args.dst, "generation_config.json")))
        for k in ("top_k", "top_p", "do_sample"):
            gc.pop(k, None)
        gc["temperature"] = args.temperature
        gc["do_sample"] = args.temperature > 0
        json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
        print("generation_config:", json.dumps(gc))

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(args.dst, f))]
    assert not missing, f"missing files in {args.dst}: {missing}"
    assert not any(
        os.path.islink(os.path.join(args.dst, f)) for f in os.listdir(args.dst)
    ), "final_model contains symlinks"

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    print("architectures:", cfg["architectures"])
    assert "gemma" in cfg["architectures"][0].lower(), "evaluate.py would not pick gemma3.jinja"

    # the grader loads with vLLM from a fresh process; a CPU transformers load is
    # the cheap proxy that catches a broken index / dtype / missing shard
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.dst)
    m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16, device_map="cpu")
    n = sum(p.numel() for p in m.parameters())
    print("loaded ok:", type(m).__name__, f"{n/1e9:.3f}B params", "dtype", next(m.parameters()).dtype)

    tok2 = load_tokenizer(args.dst)
    p = render_prompt(tok2, "What is 2+2?")
    assert p.startswith("<bos><start_of_turn>user"), p[:60]
    assert p.endswith("<start_of_turn>model\n"), p[-40:]
    print("prompt renders through templates/gemma3.jinja ok")
    print("stop token id:", tok.convert_tokens_to_ids(STOP_TOKEN))
    print("eos in generation_config:", json.load(open(os.path.join(args.dst, "generation_config.json")))["eos_token_id"])
    print("FINAL MODEL OK:", args.dst)


if __name__ == "__main__":
    main()
