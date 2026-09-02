#!/usr/bin/env python3
"""Uniformly average the weights of several checkpoints into one ("model soup").

All inputs must share the architecture and be fine-tunes of the same base, which
is the case for every checkpoint in this session.  Averaging is done in float32
and the result is written in bfloat16 with the tokenizer and a greedy
generation_config, so the output directory is directly loadable by the grader.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXTRA_FILES = ["preprocessor_config.json", "processor_config.json",
               "added_tokens.json", "tokenizer.model"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.inputs)] * len(args.inputs)
    assert len(w) == len(args.inputs)
    w = [x / sum(w) for x in w]
    print("weights:", dict(zip(args.inputs, w)), flush=True)

    acc = None
    for path, wi in zip(args.inputs, w):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        sd = m.state_dict()
        if acc is None:
            acc = {k: v.clone().mul_(wi) for k, v in sd.items()}
        else:
            assert acc.keys() == sd.keys(), "state dicts differ"
            for k in acc:
                acc[k].add_(sd[k], alpha=wi)
        del m, sd
        print("merged", path, flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.inputs[0], dtype=torch.float32)
    model.load_state_dict(acc)
    model.to(torch.bfloat16)
    model.save_pretrained(args.out, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.inputs[0])
    tok.save_pretrained(args.out)
    for fn in EXTRA_FILES:
        src = os.path.join(args.inputs[0], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))

    gc = os.path.join(args.out, "generation_config.json")
    d = json.load(open(gc))
    d["do_sample"] = False
    d["temperature"] = 0.0
    d.pop("top_k", None)
    d.pop("top_p", None)
    json.dump(d, open(gc, "w"), indent=2)
    print("[soup] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
