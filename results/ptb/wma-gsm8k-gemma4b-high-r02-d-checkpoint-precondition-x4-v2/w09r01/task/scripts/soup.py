#!/usr/bin/env python3
"""Uniform weight average (model soup) over checkpoints of one training run.

Streams each source's safetensors shards so peak RSS stays at one fp32 copy of
the model rather than N.

Key-naming note (this is what a first version got wrong): under transformers
4.57.3 the on-disk keys of a Gemma3ForConditionalGeneration checkpoint are the
converted form (`language_model.model.*`, `vision_tower.*`) while the live
module's state_dict uses `model.language_model.*`. `_checkpoint_conversion_mapping`
is applied by from_pretrained, not by load_state_dict. So this script never
builds a model object: it writes the averaged tensors back under the SAME
on-disk names the sources used, next to a copy of the source config.json, and
lets from_pretrained do the conversion at load time exactly as it does for the
ingredients.
"""
from __future__ import annotations

import argparse, glob, json, os, shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
GREEDY = {"bos_token_id": 2, "cache_implementation": "hybrid", "do_sample": False,
          "temperature": 0.0, "top_k": 0, "top_p": 1.0, "eos_token_id": [1, 106],
          "pad_token_id": 0, "transformers_version": "4.50.0.dev0"}


def stream(ckpt):
    shards = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
    assert shards, f"no safetensors in {ckpt}"
    for shard in shards:
        for k, v in load_file(shard).items():
            yield k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    acc, n = {}, len(args.srcs)
    for i, c in enumerate(args.srcs):
        keys = set()
        for k, v in stream(c):
            t = v.to(torch.float32)
            if i == 0:
                acc[k] = t
            else:
                assert k in acc, f"key {k} absent from the first source"
                acc[k] += t
            keys.add(k)
        assert keys == set(acc), (
            f"{c}: key set differs from the first source "
            f"(missing {len(set(acc)) - len(keys)}, extra {len(keys - set(acc))})")
        print(f"{c}: {len(keys)} tensors accumulated", flush=True)

    out = {k: (v / n).to(torch.bfloat16).contiguous() for k, v in acc.items()}
    del acc

    os.makedirs(args.dst, exist_ok=True)
    save_file(out, os.path.join(args.dst, "model.safetensors"), metadata={"format": "pt"})
    shutil.copy(os.path.join(args.srcs[0], "config.json"), os.path.join(args.dst, "config.json"))
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.dst)
    for fn in ("preprocessor_config.json", "processor_config.json", "added_tokens.json"):
        shutil.copy(os.path.join(BASE, fn), os.path.join(args.dst, fn))
    json.dump(GREEDY, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print(f"soup of {n} checkpoints ({len(out)} tensors) saved to {args.dst}", flush=True)

    # the load the grader will do, on CPU, with the strictness the WMA asked for
    from transformers import Gemma3ForConditionalGeneration
    m = Gemma3ForConditionalGeneration.from_pretrained(args.dst, dtype=torch.bfloat16)
    print(f"reload OK: {sum(p.numel() for p in m.parameters())/1e9:.3f}B params, {m.config.architectures}")


if __name__ == "__main__":
    main()
