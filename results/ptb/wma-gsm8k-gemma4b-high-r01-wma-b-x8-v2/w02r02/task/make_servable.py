#!/usr/bin/env python3
"""Make a Trainer checkpoint servable by vLLM, and choose its decode.

The Trainer was built without a processing_class, so checkpoint-*/ holds
weights + config only. vLLM also needs the tokenizer/processor files, and --
because the grader sends no temperature -- the checkpoint's own
generation_config.json *is* the decode configuration.

    python make_servable.py --src ckpts/exp-02/checkpoint-1866 --dst ckpts/exp-02/ep1 --decode greedy
"""
import argparse
import json
import shutil
from pathlib import Path

SNAPSHOT = Path(
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
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--decode", choices=["inherit", "greedy"], default="greedy")
    ap.add_argument("--link", action="store_true", help="hardlink weights instead of copying")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for p in src.iterdir():
        if p.name.startswith("checkpoint-") or p.is_dir():
            continue
        if p.name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            continue
        target = dst / p.name
        if target.exists():
            continue
        if args.link and p.suffix == ".safetensors":
            target.hardlink_to(p)
        else:
            shutil.copy2(p, target)

    for name in AUX:
        s = SNAPSHOT / name
        if s.exists() and not (dst / name).exists():
            shutil.copy2(s, dst / name)

    gen = json.loads((SNAPSHOT / "generation_config.json").read_text())
    if args.decode == "greedy":
        # vLLM reads this file for its default sampling params (the grader sends
        # none). top_k is removed rather than set to -1: that sentinel has been
        # observed to break a later save_pretrained.
        gen.pop("top_k", None)
        gen["do_sample"] = False
        gen["temperature"] = 0.0
        gen["top_p"] = 1.0
    (dst / "generation_config.json").write_text(json.dumps(gen, indent=2))

    assert 106 in gen["eos_token_id"], "<end_of_turn>=106 must be an eos id"
    print(f"[ok] {dst} decode={args.decode} gen={gen}")
    print(sorted(p.name for p in dst.iterdir()))


if __name__ == "__main__":
    main()
