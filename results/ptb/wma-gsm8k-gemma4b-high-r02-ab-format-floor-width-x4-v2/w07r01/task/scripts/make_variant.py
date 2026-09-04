"""Make a decode-config variant of a checkpoint without copying the weights.

generation_config.json travels with the model, and vLLM's --generation-config auto
reads temperature/top_k/top_p/repetition_penalty out of it (vllm/config/model.py
get_diff_sampling_param). It is therefore the one decode knob available when
evaluate.py is immutable. Weights are hardlinked, so a variant costs ~0 bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0, drop top_k/top_p, do_sample false")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for name in sorted(os.listdir(args.src)):
        s, d = os.path.join(args.src, name), os.path.join(args.dst, name)
        if os.path.exists(d):
            os.remove(d)
        if name.endswith(".safetensors"):
            os.link(s, d)
        else:
            shutil.copy(s, d)

    p = os.path.join(args.dst, "generation_config.json")
    cfg = json.load(open(p))
    if args.greedy:
        cfg["do_sample"] = False
        cfg["temperature"] = 0.0
        cfg.pop("top_k", None)
        cfg.pop("top_p", None)
    for key, val in (("temperature", args.temperature), ("top_p", args.top_p),
                     ("top_k", args.top_k), ("repetition_penalty", args.repetition_penalty)):
        if val is not None:
            cfg[key] = val
    json.dump(cfg, open(p, "w"), indent=2)
    print(json.dumps(cfg, indent=2))
    print("[variant]", args.dst)


if __name__ == "__main__":
    main()
