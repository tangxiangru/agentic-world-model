#!/usr/bin/env python3
"""Turn a Trainer checkpoint into a directory the grader can actually serve.

`trainer.save_model` writes config.json + safetensors and nothing else. The
grader loads the directory with vLLM from a fresh process, and config.json still
declares Gemma3ForConditionalGeneration, so vLLM builds a Gemma3Processor from
the directory: the preprocessor/processor configs have to be there even though
we never feed it an image. This script copies everything the base snapshot has
that the checkpoint does not, writes the greedy generation_config, and then
loads the result once (config + tokenizer, and optionally the full weights on
CPU) so a broken directory is found here rather than at grading time.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

from sft_common import BASE_SNAPSHOT

# files that must exist in a servable gemma-3 directory
AUX = [
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]

GREEDY_GENERATION_CONFIG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    # temperature must be present: vLLM adopts only the keys that appear in the
    # generation config's diff dict, and with none of temperature/top_p/top_k
    # present it falls back to its own t=1.0 default. top_k is omitted on
    # purpose (-1 is a vLLM sentinel that makes transformers' save_pretrained
    # raise), and do_sample=false + temperature=0.0 is greedy for vLLM.
    "do_sample": False,
    "temperature": 0.0,
    "transformers_version": "4.57.3",
}


def package(src: str, dst: str, base: str = BASE_SNAPSHOT, full_load: bool = False) -> None:
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*")):
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    copied = []
    for name in AUX:
        tgt = os.path.join(dst, name)
        if not os.path.exists(tgt):
            srcf = os.path.join(base, name)
            if os.path.exists(srcf):
                shutil.copy2(srcf, tgt)
                copied.append(name)
    with open(os.path.join(dst, "generation_config.json"), "w") as f:
        json.dump(GREEDY_GENERATION_CONFIG, f, indent=2)
    print(f"packaged {src} -> {dst}; copied from base: {copied}")

    # sanity: the pieces the grader touches
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(dst)
    tok = AutoTokenizer.from_pretrained(dst)
    print(f"  architectures={cfg.architectures} vocab={tok.vocab_size} eot={tok.convert_tokens_to_ids('<end_of_turn>')}")
    assert "gemma" in cfg.architectures[0].lower(), "evaluate.py resolves the template from this"
    gc = json.load(open(os.path.join(dst, "generation_config.json")))
    assert gc["temperature"] == 0.0 and gc["do_sample"] is False
    assert 106 in gc["eos_token_id"]
    shards = sorted(glob.glob(os.path.join(dst, "*.safetensors")))
    total = sum(os.path.getsize(s) for s in shards) / 1e9
    print(f"  {len(shards)} safetensors shards, {total:.1f} GB")
    assert total > 5.0, "weights look truncated"

    if full_load:
        import torch
        from transformers import Gemma3ForConditionalGeneration

        m = Gemma3ForConditionalGeneration.from_pretrained(dst, dtype=torch.bfloat16)
        n = sum(p.numel() for p in m.parameters())
        print(f"  CPU load ok: {n/1e9:.2f}B params")
        del m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--full-load", action="store_true")
    args = ap.parse_args()
    package(args.src, args.dst, full_load=args.full_load)


if __name__ == "__main__":
    main()
