"""Turn a trainer checkpoint into a directory the grader can load with vLLM.

  * casts weights to bf16 and re-saves them
  * copies the tokenizer / processor files from the frozen base snapshot
  * writes a generation_config.json that keeps eos_token_id [1, 106] but
    decodes GREEDILY (the harness inherits decode settings from the model dir;
    the base snapshot ships do_sample=true / top_k=64 / top_p=0.95)
  * reloads the result on CPU to prove it is loadable (pitfall
    final_model_not_loadable)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

SNAP = Path(
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
AUX = [
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
]

GREEDY = {
    "bos_token_id": 2,
    "pad_token_id": 0,
    "eos_token_id": [1, 106],
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sample", action="store_true", help="keep the base sampling config")
    args = ap.parse_args()

    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print("[load]", args.src, flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(args.src, dtype=torch.bfloat16)
    model.config.use_cache = True
    model.save_pretrained(dst, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.save_pretrained(dst)
    for f in AUX:
        p = SNAP / f
        if p.exists() and not (dst / f).exists():
            shutil.copy(p, dst / f)

    gen = dict(GREEDY)
    if args.sample:
        gen.update({"do_sample": True, "top_k": 64, "top_p": 0.95})
        gen.pop("temperature", None)
    (dst / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")

    del model
    print("[verify] reloading on CPU", flush=True)
    m2 = Gemma3ForConditionalGeneration.from_pretrained(dst, dtype=torch.bfloat16)
    n = sum(p.numel() for p in m2.parameters())
    print(f"[verify] ok, {n / 1e9:.2f}B params, arch={m2.config.architectures}", flush=True)
    print("[verify] generation_config:", (dst / "generation_config.json").read_text().strip())
    print("[done]", dst)


if __name__ == "__main__":
    main()
