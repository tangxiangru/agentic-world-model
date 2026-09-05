#!/usr/bin/env python3
"""Assemble a checkpoint into a directory the grader's vLLM can load cold.

Trainer.save_model + tokenizer.save_pretrained do not write the *processor*
side-cars, and gemma-3-4b is a Gemma3ForConditionalGeneration: vLLM builds an
image processor for it at load time. A missing preprocessor_config.json is the
final_model_not_loadable pitfall, and it only shows up in a fresh process.

Copies any missing side-car from the immutable base snapshot, optionally rewrites
generation_config.json for greedy decoding, then loads the result with
AutoConfig/AutoTokenizer/AutoProcessor to prove it resolves.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE = Path("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
            "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
SIDECARS = ["preprocessor_config.json", "processor_config.json", "added_tokens.json",
            "tokenizer.model", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="trainer output dir (…/final)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    for name in SIDECARS:
        if not (dst / name).exists() and (BASE / name).exists():
            shutil.copy2(BASE / name, dst / name)
            print(f"copied missing side-car: {name}")

    gc_path = dst / "generation_config.json"
    gc = json.loads(gc_path.read_text()) if gc_path.exists() else \
        json.loads((BASE / "generation_config.json").read_text())
    if args.greedy:
        gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    gc.setdefault("eos_token_id", [1, 106])
    gc_path.write_text(json.dumps(gc, indent=2))
    print("generation_config:", json.dumps(gc))

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    cfg = AutoConfig.from_pretrained(dst)
    print("architectures:", cfg.architectures)
    tok = AutoTokenizer.from_pretrained(dst)
    print("tokenizer ok, eot id:", tok.convert_tokens_to_ids("<end_of_turn>"))
    try:
        AutoProcessor.from_pretrained(dst)
        print("processor ok")
    except Exception as e:  # noqa: BLE001
        print("PROCESSOR FAILED:", e)
    weights = sorted(p.name for p in dst.glob("*.safetensors"))
    total = sum(p.stat().st_size for p in dst.glob("*.safetensors")) / 1e9
    print(f"{len(weights)} shards, {total:.1f} GB")


if __name__ == "__main__":
    main()
