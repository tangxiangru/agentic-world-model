"""Uniform weight average of two or more checkpoints of the same architecture."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(w) == len(args.models)
    s = sum(w)
    w = [x / s for x in w]
    print("weights", w, flush=True)

    acc = None
    for wi, m in zip(w, args.models):
        model = Gemma3ForConditionalGeneration.from_pretrained(m, dtype=torch.float32)
        sd = model.state_dict()
        if acc is None:
            acc = {k: v * wi for k, v in sd.items()}
        else:
            assert set(acc) == set(sd), "state dicts differ"
            for k in acc:
                acc[k] += sd[k] * wi
        del model, sd

    model = Gemma3ForConditionalGeneration.from_pretrained(args.models[0], dtype=torch.float32)
    model.load_state_dict(acc)
    model = model.to(torch.bfloat16)
    os.makedirs(args.out, exist_ok=True)
    # the parents carry the greedy generation_config (do_sample False + temperature 0.0),
    # which transformers' own validator rejects on save; write a valid one here and put
    # the greedy values back into the json afterwards, which is what vLLM reads.
    model.generation_config.do_sample = True
    model.generation_config.temperature = None
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.models[0]).save_pretrained(args.out)
    for extra in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.models[0], extra)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.out, extra))
    gc_path = os.path.join(args.out, "generation_config.json")
    gc = json.load(open(gc_path))
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
