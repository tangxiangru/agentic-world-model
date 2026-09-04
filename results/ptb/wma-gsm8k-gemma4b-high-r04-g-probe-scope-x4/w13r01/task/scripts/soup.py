#!/usr/bin/env python3
"""Uniform (or weighted) parameter average of two or more of our checkpoints.

All arms descend from the same google/gemma-3-4b-pt initialisation, which is the
condition under which weight averaging is well behaved. Runs on CPU so it can be
prepared while the GPU is busy.

The output directory is a complete, self-contained model: tokenizer, the greedy
generation_config.json the grader's vLLM reads, and the preprocessor/processor
configs the Gemma3ForConditionalGeneration architecture expects.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def shard_files(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return sorted(set(wm.values())), idx
    return ["model.safetensors"], None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ws = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(ws) == len(args.models), "one weight per model"
    tot = sum(ws)
    ws = [w / tot for w in ws]
    print("averaging", list(zip(args.models, [round(w, 4) for w in ws])))

    os.makedirs(args.out, exist_ok=True)
    files, idx = shard_files(args.models[0])

    n_params = 0
    for fn in files:
        acc = None
        for m, w in zip(args.models, ws):
            sd = load_file(os.path.join(m, fn))
            if acc is None:
                acc = {k: v.to(torch.float32) * w for k, v in sd.items()}
            else:
                assert set(sd) == set(acc), f"key mismatch in {fn} for {m}"
                for k, v in sd.items():
                    acc[k].add_(v.to(torch.float32), alpha=w)
            del sd
        acc = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        n_params += sum(v.numel() for v in acc.values())
        save_file(acc, os.path.join(args.out, fn), metadata={"format": "pt"})
        print("wrote", fn, flush=True)
        del acc

    src = args.models[0]
    for f in ("config.json", "model.safetensors.index.json", "tokenizer.json",
              "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
              "added_tokens.json", "preprocessor_config.json",
              "processor_config.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, f))

    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "do_sample": False, "temperature": 0.0, "top_p": 1.0,
                   "cache_implementation": "hybrid"}, f, indent=2)

    print(f"soup written to {args.out} ({n_params/1e9:.3f}B params)")


if __name__ == "__main__":
    main()
