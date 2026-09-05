"""Turn a raw Trainer checkpoint-<step>/ into a directory vLLM can serve.

The Trainer writes fp32 weights and none of the multimodal sidecar files, so a
checkpoint straight out of ckpts/*/checkpoint-N is not what the grader loads.
This casts to bf16, copies the processor configs and the tokenizer, and writes
a greedy generation_config.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_CONFIG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "cache_implementation": "hybrid",
}


def finalize(src: str, dst: str, base: str) -> None:
    os.makedirs(dst, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    model.config.use_cache = True
    model.save_pretrained(dst, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base)
    tok.save_pretrained(dst)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        p = os.path.join(base, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, fn))
    with open(os.path.join(dst, "generation_config.json"), "w") as f:
        json.dump(GEN_CONFIG, f, indent=2)
    print(f"finalized {src} -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--base", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    a = ap.parse_args()
    finalize(a.src, a.dst, a.base)
