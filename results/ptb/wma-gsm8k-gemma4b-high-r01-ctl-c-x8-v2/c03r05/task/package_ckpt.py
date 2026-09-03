#!/usr/bin/env python3
"""Turn a raw Trainer checkpoint into something the grader's vLLM can serve.

Copies the small config/tokenizer files and symlinks the safetensors, then
writes the tokenizer with eos=<end_of_turn> and a greedy generation_config.json
-- the same treatment train_sft.save_for_grader gives the final checkpoint.
"""
import argparse
import json
import os
import shutil

from transformers import AutoTokenizer

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
END_OF_TURN = "<end_of_turn>"


def package(src: str, dst: str, copy: bool = False) -> None:
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        if fn.endswith(".safetensors"):
            d = os.path.join(dst, fn)
            if os.path.lexists(d):
                os.remove(d)
            (shutil.copy2 if copy else os.symlink)(os.path.abspath(os.path.join(src, fn)), d)
        elif fn in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    for fn in ("preprocessor_config.json", "processor_config.json"):
        p = os.path.join(BASE, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, fn))
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("templates/gemma3.jinja").read()
    tok.eos_token = END_OF_TURN
    tok.save_pretrained(dst)
    shutil.copy2(os.path.join(BASE, "tokenizer.json"), os.path.join(dst, "tokenizer.json"))
    with open(os.path.join(dst, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "do_sample": False, "temperature": 0.0,
                   "cache_implementation": "hybrid"}, f, indent=2)
    print("packaged", src, "->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--copy", action="store_true")
    a = ap.parse_args()
    package(a.src, a.dst, a.copy)
