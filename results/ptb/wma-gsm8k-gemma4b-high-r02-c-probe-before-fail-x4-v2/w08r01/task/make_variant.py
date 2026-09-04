"""Make a decode-only variant of a checkpoint: same weights, different generation_config.

Weights are symlinked, so this costs no disk and no copy time. vLLM reads
generation_config.json for its default sampling params (evaluate.py sends no
temperature, so those defaults are what the grader actually decodes with).
"""
from __future__ import annotations

import argparse
import json
import os

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    src = os.path.abspath(args.src)
    for f in os.listdir(src):
        if f in ("generation_config.json",):
            continue
        if f.startswith(("optimizer", "scheduler", "rng_state", "trainer_state", "training_args")):
            continue
        d = os.path.join(args.dst, f)
        if os.path.lexists(d):
            os.remove(d)
        os.symlink(os.path.join(src, f), d)

    gc = json.load(open(os.path.join(src, "generation_config.json")))
    eos = gc.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    assert 106 in eos, f"stop token <end_of_turn> missing from eos_token_id: {eos}"
    if args.temperature is not None:
        gc["temperature"] = args.temperature
        gc["do_sample"] = args.temperature > 0
        if args.temperature == 0:
            gc.pop("top_k", None)
            gc.pop("top_p", None)
    if args.top_p is not None:
        gc["top_p"] = args.top_p
    if args.top_k is not None:
        gc["top_k"] = args.top_k
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print(args.dst, "->", json.dumps(gc))


if __name__ == "__main__":
    main()
