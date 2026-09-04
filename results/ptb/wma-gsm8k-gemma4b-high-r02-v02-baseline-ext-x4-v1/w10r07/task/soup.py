#!/usr/bin/env python3
"""Uniformly average the weights of several checkpoints fine-tuned from the same init.

All inputs must descend from the same google/gemma-3-4b-pt snapshot, so the
parameters live in one basin and a plain average is meaningful.  Averaging is
done in fp32 and the result is written bf16 with the greedy generation_config
and the snapshot's verbatim tokenizer files (see prep_ckpt.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--base", default=os.environ.get("PTB_BASE_MODEL_SNAPSHOT"))
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    from transformers import Gemma3ForConditionalGeneration

    acc = None
    for i, src in enumerate(args.src):
        m = Gemma3ForConditionalGeneration.from_pretrained(src, dtype=torch.float32)
        sd = m.state_dict()
        if acc is None:
            acc = {k: v.clone() for k, v in sd.items()}
        else:
            assert set(acc) == set(sd), "state dicts differ"
            for k in acc:
                acc[k] += sd[k]
        del m, sd
        print(f"[soup] accumulated {i+1}/{len(args.src)}: {src}", flush=True)

    n = len(args.src)
    for k in acc:
        acc[k] = (acc[k] / n).to(torch.bfloat16)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.src[0], dtype=torch.bfloat16
    )
    missing, unexpected = model.load_state_dict(acc, strict=False)
    assert not unexpected, unexpected
    print("[soup] missing keys (tied weights are expected here):", missing)
    model.config.torch_dtype = "bfloat16"
    model.config.use_cache = True
    os.makedirs(args.dst, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)

    for fname in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        src = os.path.join(args.base, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.dst, fname))

    gen = json.load(open(os.path.join(args.base, "generation_config.json")))
    if args.greedy:
        gen.pop("top_k", None)
        gen.pop("top_p", None)
        gen["do_sample"] = False
        gen["temperature"] = 0.0
    json.dump(gen, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print("[soup] wrote", args.dst, gen)


if __name__ == "__main__":
    main()
