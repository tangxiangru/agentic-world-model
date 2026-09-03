#!/usr/bin/env python3
"""Write a decode config into a checkpoint directory.

evaluate.py never sets a temperature, so vLLM falls back to the model's own
generation_config.json ("Default sampling parameters have been overridden by the
model's Hugging Face generation config" in the server log). vLLM reads exactly
these keys from it: repetition_penalty, temperature, top_k, top_p, min_p,
max_new_tokens. The base snapshot ships do_sample=true / top_k=64 / top_p=0.95,
i.e. the benchmark is graded on a temperature-1.0 sample.

  greedy  -> temperature 0.0, no top_k/top_p  (vLLM decodes greedily)
  sampled -> the base snapshot's own values   (the default the run inherits)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = {"bos_token_id": 2, "cache_implementation": "hybrid",
        "eos_token_id": [1, 106], "pad_token_id": 0}

MODES = {
    "greedy": {**BASE, "do_sample": False, "temperature": 0.0},
    "sampled": {**BASE, "do_sample": True, "top_k": 64, "top_p": 0.95},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--mode", choices=sorted(MODES), default="greedy")
    a = ap.parse_args()
    p = Path(a.ckpt) / "generation_config.json"
    p.write_text(json.dumps(MODES[a.mode], indent=2) + "\n")
    print(f"{p}: {a.mode} -> {json.dumps(MODES[a.mode])}")


if __name__ == "__main__":
    main()
