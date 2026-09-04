#!/usr/bin/env python3
"""Write final_model/ from a training checkpoint: bf16 weights, tokenizer,
processor files, and a generation_config the grader's vLLM will honour.

evaluate.py passes no sampling parameters, so vLLM takes temperature / top_p /
top_k from generation_config.json. `--temperature` writes that field; leaving it
out keeps the base checkpoint's decode (no temperature key -> vLLM default 1.0).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)

    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16)
    model.save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.dst)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        shutil.copy(os.path.join(BASE, fn), os.path.join(args.dst, fn))

    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    if args.temperature is not None:
        gc["temperature"] = args.temperature
        gc["do_sample"] = args.temperature > 0
    if args.top_p is not None:
        gc["top_p"] = args.top_p
    if args.top_k is not None:
        gc["top_k"] = args.top_k
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print("generation_config:", gc)

    # the pitfall this guards: a final_model the grader cannot load
    del model
    m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(args.dst)
    print("[verify] reloaded", type(m).__name__, sum(p.numel() for p in m.parameters()) / 1e9, "B params")
    print("[verify] tokenizer eos", tk.eos_token, "| files:", sorted(os.listdir(args.dst)))
    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    assert cfg["architectures"] == ["Gemma3ForConditionalGeneration"], cfg["architectures"]
    g2 = json.load(open(os.path.join(args.dst, "generation_config.json")))
    assert g2["eos_token_id"] == [1, 106], g2
    print("[verify] ok")


if __name__ == "__main__":
    main()
