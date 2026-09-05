#!/usr/bin/env python3
"""Turn a Trainer checkpoint into a directory the grader's vLLM can load.

evaluate.py builds the model path itself and reads nothing but the directory,
so the directory has to carry: weights, config, tokenizer, the multimodal
processor configs gemma-3 needs, and a generation_config.

vLLM reads only temperature / top_k / top_p / min_p / repetition_penalty out of
generation_config.json (vllm/config/model.py get_diff_sampling_param); do_sample
is ignored. So `--decode greedy` is expressed as temperature 0.0 with the
sampling knobs removed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE = Path(
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
AUX = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkpoint dir with the weights")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--decode", choices=["base", "greedy"], default="greedy")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    model = Gemma3ForConditionalGeneration.from_pretrained(
        src, dtype=getattr(torch, args.dtype)
    )
    model.config.use_cache = True
    model.save_pretrained(dst, safe_serialization=True)

    try:
        tok = AutoTokenizer.from_pretrained(src)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE)
    tok.save_pretrained(dst)
    for name in AUX:
        p = src / name
        if not p.exists():
            p = BASE / name
        if p.exists():
            shutil.copy2(p, dst / name)

    gen = json.loads((BASE / "generation_config.json").read_text())
    if args.decode == "greedy":
        gen["do_sample"] = False
        gen["temperature"] = 0.0
        gen.pop("top_k", None)
        gen.pop("top_p", None)
    (dst / "generation_config.json").write_text(json.dumps(gen, indent=2))
    print(json.dumps(gen, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
