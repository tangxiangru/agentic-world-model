#!/usr/bin/env python3
"""Turn a Trainer checkpoint into a directory the grader's vLLM can load.

Three things the raw checkpoint is missing or wrong about:
  * it is saved in fp32 (the run keeps fp32 master weights), so vLLM would serve fp32
  * it has no tokenizer and no processor/preprocessor config, which Gemma3's
    multimodal wrapper needs
  * its generation_config still carries gemma's sampling defaults (do_sample true,
    top_k 64, top_p 0.95); evaluate.py sets no temperature, so those defaults decide
    whether the benchmark read is sampled or greedy
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
AUX = [
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", choices=["inherit", "greedy"], default="inherit")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16)
    model.save_pretrained(out, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(out)
    for name in AUX:
        src = Path(args.base) / name
        if src.exists() and not (out / name).exists():
            shutil.copy(src, out / name)

    gc_path = out / "generation_config.json"
    gc = json.load(open(gc_path)) if gc_path.exists() else {}
    gc["eos_token_id"] = [1, 106]     # <eos> and <end_of_turn>, the terminator we trained on
    gc["bos_token_id"] = 2
    gc["pad_token_id"] = 0
    if args.decode == "greedy":
        # vLLM's get_diff_sampling_param reads temperature/top_k/top_p out of this file
        # and there is no --temperature on evaluate.py, so this file is the decode config
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc["top_k"] = 0
        gc["top_p"] = 1.0
    json.dump(gc, open(gc_path, "w"), indent=2)

    cfg = json.load(open(out / "config.json"))
    print("packaged", out, "dtype:", cfg.get("dtype") or cfg.get("torch_dtype"))
    print("generation_config:", json.dumps(gc))
    print("files:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
