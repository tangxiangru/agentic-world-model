#!/usr/bin/env python3
"""Trainer's intermediate checkpoints hold weights + config only. vLLM also
needs the tokenizer, the processor configs and a generation config: copy them
from the base snapshot so an epoch checkpoint is loadable as-is."""
from __future__ import annotations

import argparse
import os
import shutil

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
FILES = [
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "preprocessor_config.json", "processor_config.json",
    "generation_config.json",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    a = ap.parse_args()
    for fn in FILES:
        src = os.path.join(BASE, fn)
        dst = os.path.join(a.ckpt, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print("copied", fn)
    print("ok", a.ckpt)
