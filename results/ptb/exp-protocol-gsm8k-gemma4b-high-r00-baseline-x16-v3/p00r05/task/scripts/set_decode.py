#!/usr/bin/env python3
"""Rewrite a model dir's generation_config.json to pin the decoding the grader will use.

evaluate.py never sends a temperature: inspect_ai only forwards fields that are
set on GenerateConfig, and it sets none. vLLM therefore falls back to
`--generation-config auto`, i.e. ModelConfig.get_diff_sampling_param(), which
reads temperature / top_p / top_k / min_p / repetition_penalty out of the model
directory's generation_config.json (vllm/config/model.py:1344).

The base snapshot ships do_sample=true, top_k=64, top_p=0.95 and no temperature,
so the harness samples at temperature 1.0. Writing temperature 0.0 and dropping
top_k/top_p makes the graded decode greedy.
"""
from __future__ import annotations

import argparse
import json
import os

PRESETS = {
    "greedy": {"do_sample": False, "temperature": 0.0},
    "base": {"do_sample": True, "top_k": 64, "top_p": 0.95},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="greedy")
    args = ap.parse_args()

    cfg = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
    }
    cfg.update(PRESETS[args.preset])
    path = os.path.join(args.model_dir, "generation_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", path)
    print(json.dumps(cfg, indent=2))

    # show exactly what vLLM will pick up
    from transformers import GenerationConfig

    gc = GenerationConfig.from_pretrained(args.model_dir)
    diff = gc.to_diff_dict()
    avail = ["repetition_penalty", "temperature", "top_k", "top_p", "min_p", "max_new_tokens"]
    print("vLLM default sampling params ->", {k: diff[k] for k in avail if k in diff})


if __name__ == "__main__":
    main()
