#!/usr/bin/env python3
"""Trainer saves intermediate checkpoints without tokenizer files (no
processing_class was passed), so vLLM cannot serve them. Copy the tokenizer /
processor files from the immutable base snapshot into a checkpoint dir."""
import os, shutil, sys
SNAP="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
FILES=["tokenizer.json","tokenizer.model","tokenizer_config.json","special_tokens_map.json",
       "added_tokens.json","preprocessor_config.json","processor_config.json"]
for ck in sys.argv[1:]:
    for f in FILES:
        s=os.path.join(SNAP,f); d=os.path.join(ck,f)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy(s,d)
    print("fixed",ck, sorted(os.listdir(ck)))
