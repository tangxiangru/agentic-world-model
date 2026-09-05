#!/usr/bin/env python3
"""Uniform weight average (model soup) of two checkpoints of the same architecture."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

TOK_FILES = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json",
             "preprocessor_config.json", "processor_config.json"]


def state_dict_of(path: str) -> dict[str, torch.Tensor]:
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    sd: dict[str, torch.Tensor] = {}
    for shard in sorted(set(idx["weight_map"].values())):
        sd.update(load_file(os.path.join(path, shard)))
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sa, sb = state_dict_of(args.a), state_dict_of(args.b)
    assert set(sa) == set(sb), (set(sa) ^ set(sb))
    ndiff = 0
    for k in sa:
        if not torch.equal(sa[k], sb[k]):
            ndiff += 1
        sa[k] = (args.alpha * sa[k].float()
                 + (1 - args.alpha) * sb[k].float()).to(sa[k].dtype)
    print(f"averaged {len(sa)} tensors, {ndiff} differed between the two checkpoints",
          flush=True)
    del sb

    # Rewrite the shards in place, keeping --a's exact file layout and key names,
    # so no state-dict key remapping is involved.
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    shutil.copytree(args.a, args.out)
    idx = json.load(open(os.path.join(args.out, "model.safetensors.index.json")))
    for shard in sorted(set(idx["weight_map"].values())):
        keys = [k for k, v in idx["weight_map"].items() if v == shard]
        save_file({k: sa[k].contiguous() for k in keys},
                  os.path.join(args.out, shard), metadata={"format": "pt"})
        print("wrote", shard, len(keys), "tensors", flush=True)
    with open(os.path.join(args.out, "generation_config.json"), "w") as fh:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "cache_implementation": "hybrid", "do_sample": False,
                   "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                   "transformers_version": "4.57.3"}, fh, indent=2)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
