"""Uniform weight average of checkpoints that share a parent (model soup).

Averages every tensor of the safetensors shards; non-float tensors are taken
from the first model. Writes a full loadable checkpoint (config, tokenizer,
greedy generation_config) so it can be graded exactly like any other.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(w) == len(args.models)
    s = sum(w)
    w = [x / s for x in w]
    print("weights:", dict(zip(args.models, w)))

    os.makedirs(args.out, exist_ok=True)
    idx_path = os.path.join(args.models[0], "model.safetensors.index.json")
    index = json.load(open(idx_path))
    shards = sorted(set(index["weight_map"].values()))

    for sh in shards:
        acc = None
        for m, wi in zip(args.models, w):
            sd = load_file(os.path.join(m, sh))
            if acc is None:
                acc = {k: (v.to(torch.float32) * wi if v.is_floating_point() else v.clone())
                       for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    if v.is_floating_point():
                        acc[k] += v.to(torch.float32) * wi
            del sd
        out = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in acc.items()}
        save_file(out, os.path.join(args.out, sh), metadata={"format": "pt"})
        print("wrote", sh, len(out), "tensors", flush=True)
        del acc, out

    for fn in ("config.json", "model.safetensors.index.json", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
               "added_tokens.json", "preprocessor_config.json", "processor_config.json",
               "generation_config.json"):
        src = os.path.join(args.models[0], fn)
        if not os.path.exists(src):
            src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))

    gc_path = os.path.join(args.out, "generation_config.json")
    gc = json.load(open(gc_path))
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("[done]", args.out, sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
