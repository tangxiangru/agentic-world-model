#!/usr/bin/env python3
"""Write an explicit decode arm into a checkpoint's generation_config.json.

vLLM 0.11 reads generation_config.json from the model directory and uses it as
the server's default sampling params, so this file - not evaluate.py - decides
the decode arm the grader sees. gemma-3-pt ships do_sample=true, top_k=64,
top_p=0.95 and no temperature, i.e. sampling at T=1.0; those survive
save_pretrained and must be removed explicitly to decode greedily.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["greedy", "sampling"], required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    path = os.path.join(args.ckpt, "generation_config.json")
    gc = json.load(open(path))
    if not os.path.exists(path + ".orig"):
        shutil.copy(path, path + ".orig")

    if args.mode == "greedy":
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_p", None)
        gc.pop("top_k", None)
    else:
        gc["do_sample"] = True
        gc["temperature"] = args.temperature if args.temperature is not None else 1.0
        if args.top_p is not None:
            gc["top_p"] = args.top_p
        if args.top_k is not None:
            gc["top_k"] = args.top_k

    eos = gc.get("eos_token_id")
    if isinstance(eos, int):
        eos = [eos]
    if eos is None or 106 not in eos:
        gc["eos_token_id"] = sorted(set((eos or []) + [1, 106]))
    json.dump(gc, open(path, "w"), indent=2)
    print(json.dumps(gc, indent=2))


if __name__ == "__main__":
    main()
