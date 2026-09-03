#!/usr/bin/env python3
"""Make a decode-config variant of a checkpoint: symlink the weights, rewrite
generation_config.json.

vLLM reads temperature/top_k/top_p/min_p/repetition_penalty out of
generation_config.json (ModelConfig.get_diff_sampling_param) and ignores
do_sample, so greedy decoding has to be expressed as temperature 0.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

BIG = (".safetensors", ".bin", ".pt", ".model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy weights instead of symlinking")
    ap.add_argument("--extra-from", default=None, help="dir to pull missing small files (tokenizer, processor) from")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.src):
        s, d = os.path.join(args.src, name), os.path.join(args.dst, name)
        if os.path.exists(d) or os.path.islink(d):
            continue
        if name.endswith(BIG) and not args.copy:
            os.symlink(os.path.realpath(s), d)
        elif os.path.isdir(s):
            continue
        else:
            shutil.copy2(s, d)

    if args.extra_from:
        for name in os.listdir(args.extra_from):
            s2, d2 = os.path.join(args.extra_from, name), os.path.join(args.dst, name)
            if os.path.exists(d2) or os.path.islink(d2) or os.path.isdir(s2):
                continue
            if name.endswith((".safetensors", ".bin", ".pt")) or name == "model.safetensors.index.json":
                continue
            shutil.copy2(s2, d2)

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    # transformers can collapse gemma-3's eos_token_id [1, 106] to 1 on save; if
    # <end_of_turn> (106) is missing, vLLM never stops on the token the model was
    # trained to emit and every generation runs to the token cap.
    eos = gc.get("eos_token_id")
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    for t in (1, 106):
        if t not in eos:
            eos.append(t)
    gc["eos_token_id"] = eos
    gc["do_sample"] = args.temperature > 0
    gc["temperature"] = args.temperature
    gc["top_p"] = args.top_p
    gc["top_k"] = args.top_k
    json.dump(gc, open(gc_path, "w"), indent=2)
    print(json.dumps(gc, indent=2))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
