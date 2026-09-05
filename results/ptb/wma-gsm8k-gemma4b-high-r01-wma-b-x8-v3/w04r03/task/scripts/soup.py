#!/usr/bin/env python3
"""Uniformly average the weights of several checkpoints of the same lineage."""
from __future__ import annotations

import argparse
import json
import os
import shutil

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

BASE = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
GEN_CFG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "transformers_version": "4.50.0.dev0",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    acc = None
    for i, s in enumerate(a.srcs):
        m = AutoModelForCausalLM.from_pretrained(s, dtype=torch.float32)
        sd = m.state_dict()
        if acc is None:
            acc = {k: v.clone() for k, v in sd.items()}
        else:
            assert set(acc) == set(sd), "state dicts differ"
            for k in acc:
                acc[k] += sd[k]
        del m, sd
        print("added", s, flush=True)
    n = len(a.srcs)
    for k in acc:
        acc[k] = (acc[k] / n).to(torch.bfloat16)

    m = AutoModelForCausalLM.from_pretrained(a.srcs[0], dtype=torch.bfloat16)
    missing, unexpected = m.load_state_dict(acc, strict=False)
    print("missing", len(missing), "unexpected", len(unexpected))
    m.config.use_cache = True
    m.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=False,
    )
    os.makedirs(a.dst, exist_ok=True)
    m.save_pretrained(a.dst, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("/home/ben/task/templates/gemma3.jinja").read()
    tok.save_pretrained(a.dst)
    for fn in ["preprocessor_config.json", "processor_config.json", "tokenizer.model"]:
        p = os.path.join(BASE, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(a.dst, fn))
    json.dump(GEN_CFG, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
    print("souped", n, "checkpoints ->", a.dst)


if __name__ == "__main__":
    main()
