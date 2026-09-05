#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ and prove it loads the way the grader loads it.

pitfalls.yaml final_model_not_loadable: the grader starts a fresh process and
hands final_model/ to vLLM. This script makes real copies (no symlinks), checks
every file the loader needs is present, loads the config + tokenizer, renders
one prompt through templates/gemma3.jinja, and prints the decode defaults vLLM
will pick up from generation_config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalfmt as E  # noqa: E402

NEEDED = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if not args.check_only:
        if os.path.exists(args.dst):
            shutil.rmtree(args.dst)
        os.makedirs(args.dst)
        for name in sorted(os.listdir(args.src)):
            if name in ("training_args.bin",):
                continue
            src = os.path.join(args.src, name)
            if os.path.isdir(src):
                continue
            shutil.copyfile(src, os.path.join(args.dst, name))  # resolves symlinks
        print(f"copied {args.src} -> {args.dst}")

    missing = [n for n in NEEDED if not os.path.exists(os.path.join(args.dst, n))]
    if missing:
        raise SystemExit(f"FAIL missing files in {args.dst}: {missing}")

    idx = json.load(open(os.path.join(args.dst, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    for s in shards:
        p = os.path.join(args.dst, s)
        if not os.path.exists(p) or os.path.islink(p):
            raise SystemExit(f"FAIL shard missing or is a symlink: {p}")
    total = sum(os.path.getsize(os.path.join(args.dst, s)) for s in shards)
    print(f"shards: {shards}, {total/1e9:.1f} GB, all real files")

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0]
    print("architectures:", arch)
    assert "gemma" in arch.lower(), "evaluate.py routes on 'gemma' in the architecture name"

    gc = json.load(open(os.path.join(args.dst, "generation_config.json")))
    print("generation_config:", json.dumps(gc))
    assert 106 in (gc.get("eos_token_id") or []), "vLLM must stop on <end_of_turn> (106)"

    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(args.dst)
    tok = AutoTokenizer.from_pretrained(args.dst)
    tok.chat_template = E.chat_template()
    prompt = tok.apply_chat_template(
        E.messages("A shop sells 3 pens for $2 each. How much is that?", E.fewshot_system_message()),
        tokenize=False,
        add_generation_prompt=True,
    )
    n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    assert prompt.endswith("<start_of_turn>model\n"), prompt[-60:]
    print(f"grading prompt renders, {n} tokens, ends with the model turn marker")
    print("OK")


if __name__ == "__main__":
    main()
