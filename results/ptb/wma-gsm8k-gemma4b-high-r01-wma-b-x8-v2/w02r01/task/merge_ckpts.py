#!/usr/bin/env python3
"""Uniform weight average ("model soup") of two checkpoints that share a basin.

exp-05 is a warm-started continuation of exp-04, so their weights are on one
trajectory and a plain parameter mean is well defined. Output is written in bf16
with the tokenizer, processor configs and the greedy generation_config, i.e. the
same shape package_ckpt.py produces, so the grader can load it directly.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from package_ckpt import AUX, BASE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.ckpts)] * len(args.ckpts)
    assert len(w) == len(args.ckpts), "one weight per checkpoint"
    total = sum(w)
    w = [x / total for x in w]
    print("soup weights:", dict(zip(args.ckpts, w)))

    model = Gemma3ForConditionalGeneration.from_pretrained(args.ckpts[0], torch_dtype=torch.float32)
    acc = {k: v.detach().float() * w[0] for k, v in model.state_dict().items()}
    for path, wi in zip(args.ckpts[1:], w[1:]):
        other = Gemma3ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32)
        sd = other.state_dict()
        assert sd.keys() == acc.keys(), "checkpoints have different parameter sets"
        for k in acc:
            acc[k] += sd[k].detach().float() * wi
        del other, sd
    model.load_state_dict(acc)
    model = model.to(torch.bfloat16)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model.save_pretrained(out, safe_serialization=True)

    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    for name in AUX:
        src = Path(args.base) / name
        if src.exists() and not (out / name).exists():
            shutil.copy(src, out / name)

    gc_path = out / "generation_config.json"
    gc = json.load(open(gc_path)) if gc_path.exists() else {}
    gc.update({"eos_token_id": [1, 106], "bos_token_id": 2, "pad_token_id": 0,
               "do_sample": False, "temperature": 0.0, "top_k": 0, "top_p": 1.0})
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("wrote", out, json.dumps(gc))


if __name__ == "__main__":
    main()
