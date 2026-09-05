#!/usr/bin/env python3
"""Uniform weight average ("model soup") of two or more fine-tunes of the same
pretrained checkpoint.

All inputs descend from the identical base snapshot and share parameter names
and shapes, so the average is taken tensor by tensor in float32 and cast back to
bfloat16. The tokenizer, processor files and the greedy generation_config.json
are copied from the first input so the output directory loads in the grader's
vLLM exactly like its parents.
"""
import argparse
import json
import os
import shutil

import torch
from transformers import AutoTokenizer, GenerationConfig, Gemma3ForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acc = None
    for i, m in enumerate(args.models):
        print(f"loading {m}", flush=True)
        sd = Gemma3ForConditionalGeneration.from_pretrained(
            m, dtype=torch.float32, device_map="cpu").state_dict()
        if acc is None:
            acc = {k: v.clone() for k, v in sd.items()}
        else:
            assert set(acc) == set(sd), "parameter sets differ between checkpoints"
            for k in acc:
                acc[k] += sd[k]
        del sd

    n = len(args.models)
    for k in acc:
        acc[k] = (acc[k] / n).to(torch.bfloat16)

    print(f"averaged {len(acc)} tensors over {n} checkpoints", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.models[0], dtype=torch.bfloat16, device_map="cpu")
    missing, unexpected = model.load_state_dict(acc, strict=False)
    assert not [k for k in missing if "rotary" not in k], missing
    # same trap exp-03 hit: save_pretrained validates model.generation_config
    # strictly and the greedy config the parents carry (do_sample false +
    # temperature 0.0) is what it rejects. Hand it a valid one; the greedy file
    # is copied in below and is what the grader's vLLM reads.
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid")
    model.save_pretrained(args.out, safe_serialization=True)

    AutoTokenizer.from_pretrained(args.models[0]).save_pretrained(args.out)
    for fn in ("generation_config.json", "preprocessor_config.json",
               "processor_config.json"):
        src = os.path.join(args.models[0], fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.out, fn))
    print("saved", args.out, flush=True)
    print(json.load(open(os.path.join(args.out, "generation_config.json"))))


if __name__ == "__main__":
    main()
