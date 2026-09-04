"""Uniform weight average (model soup) of two or more checkpoints fine-tuned
from the same base snapshot.  CPU-only; writes a directory the grader can load.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(w) == len(args.models)
    s = sum(w)
    w = [x / s for x in w]
    print("weights:", dict(zip(args.models, w)), flush=True)

    base = Gemma3ForConditionalGeneration.from_pretrained(args.models[0], dtype=torch.float32)
    sd = base.state_dict()
    for k in sd:
        sd[k] = sd[k] * w[0]
    for m, wi in zip(args.models[1:], w[1:]):
        other = Gemma3ForConditionalGeneration.from_pretrained(m, dtype=torch.float32)
        osd = other.state_dict()
        for k in sd:
            sd[k] += osd[k] * wi
        del other, osd
        print("merged", m, flush=True)
    for k in sd:
        sd[k] = sd[k].to(torch.bfloat16)
    base.load_state_dict(sd)
    base = base.to(torch.bfloat16)
    # the checkpoints carry a greedy generation_config that transformers refuses to
    # re-serialise (temperature with do_sample=False); write it by hand after saving
    from transformers import GenerationConfig

    base.generation_config = GenerationConfig.from_pretrained(SNAPSHOT)

    out = args.out
    os.makedirs(out, exist_ok=True)
    for fn in os.listdir(SNAPSHOT):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = os.path.join(SNAPSHOT, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, fn))
    base.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(SNAPSHOT).save_pretrained(out)
    AutoTokenizer.from_pretrained(SNAPSHOT).save_pretrained(out)
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "cache_implementation": "hybrid",
    }
    json.dump(gc, open(os.path.join(out, "generation_config.json"), "w"), indent=2)
    print("[save] wrote", out, flush=True)


if __name__ == "__main__":
    main()
