"""Uniform weight average ("model soup") of two or more checkpoints of one run.

    python make_soup.py --out ckpts/soup-ep1ep2 ckpts/exp-02/checkpoint-2342 ckpts/exp-02/final

Averages every floating-point tensor on CPU in fp32 and writes a checkpoint the
grader can load, then leaves generation_config/tokenizer packaging to
finalize.py. Runs on CPU so it can share the box with a training job.
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acc = None
    for i, path in enumerate(args.ckpts):
        print("loading", path, flush=True)
        sd = Gemma3ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).state_dict()
        if acc is None:
            acc = {k: v.clone() for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k]
        del sd
    for k in acc:
        acc[k] /= len(args.ckpts)

    model = Gemma3ForConditionalGeneration.from_pretrained(args.ckpts[0], dtype=torch.float32)
    model.load_state_dict(acc)
    model = model.to(torch.bfloat16)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(SNAP).save_pretrained(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
