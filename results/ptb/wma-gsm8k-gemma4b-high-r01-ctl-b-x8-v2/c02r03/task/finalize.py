#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ and verify the grader can actually use it.

Checks, in the order they would bite:
  1. every file the grader's vLLM needs is present
  2. config.json names a gemma architecture (evaluate.py routes the template on it)
     and declares bfloat16
  3. generation_config.json requests greedy decoding and stops on <end_of_turn>
  4. the tokenizer round-trips the grader's own chat template
  5. transformers loads the weights from disk in a fresh process
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

NEEDED = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        if os.path.exists(args.dst):
            print(f"removing existing {args.dst}")
            shutil.rmtree(args.dst)
        print(f"copying {args.src} -> {args.dst}")
        shutil.copytree(args.src, args.dst)
        for junk in ("training_args.bin",):
            p = os.path.join(args.dst, junk)
            if os.path.exists(p):
                os.remove(p)

    ok = True
    files = set(os.listdir(args.dst))
    for f in NEEDED:
        if f not in files:
            print(f"FAIL missing {f}")
            ok = False
    if not any(f.endswith(".safetensors") for f in files):
        print("FAIL no safetensors weights")
        ok = False

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0]
    print(f"architecture={arch} dtype={cfg.get('dtype') or cfg.get('torch_dtype')}")
    if "gemma" not in arch.lower():
        print("FAIL evaluate.py routes the chat template on the architecture name")
        ok = False
    if (cfg.get("dtype") or cfg.get("torch_dtype")) != "bfloat16":
        print("WARN weights are not bfloat16")

    gc = json.load(open(os.path.join(args.dst, "generation_config.json")))
    print("generation_config:", gc)
    if gc.get("temperature") not in (0, 0.0) or gc.get("do_sample") is not False:
        print("FAIL generation_config does not request greedy decoding")
        ok = False
    if 106 not in (gc.get("eos_token_id") or []):
        print("FAIL <end_of_turn> (106) is not an eos id, so vLLM will not stop on it")
        ok = False

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.dst)
    tok.chat_template = open(TEMPLATE).read()
    rendered = tok.apply_chat_template(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print("rendered:", repr(rendered))
    if not rendered.endswith("<start_of_turn>model\n"):
        print("FAIL chat template does not end with the model generation prompt")
        ok = False

    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded {type(m).__name__} with {n/1e9:.2f}B params, dtype {next(m.parameters()).dtype}")

    print("OK" if ok else "PROBLEMS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
