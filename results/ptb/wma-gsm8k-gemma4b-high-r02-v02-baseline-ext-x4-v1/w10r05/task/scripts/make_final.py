"""Assemble final_model/ from a trained checkpoint and verify it is loadable.

Checks the two things that turn a good checkpoint into a zero score:
  * config.json architectures[0] must contain 'Gemma3' -- evaluate.py's
    model_type() reads it to pick templates/gemma3.jinja
  * generation_config.json eos_token_id must still contain 106 (<end_of_turn>),
    the token vLLM stops on
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

NEEDED = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--greedy", action="store_true", help="write a greedy generation_config")
    ap.add_argument(
        "--fill-from",
        default=os.environ.get("PTB_BASE_MODEL_SNAPSHOT"),
        help="copy tokenizer/processor files missing from --src (mid-run checkpoint-*/ dirs have none)",
    )
    ap.add_argument("--verify-load", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)
    for junk in ("training_args.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"):
        p = os.path.join(args.dst, junk)
        if os.path.exists(p):
            os.remove(p)

    if args.fill_from:
        for fn in NEEDED + ["tokenizer.model", "added_tokens.json"]:
            src = os.path.join(args.fill_from, fn)
            dst = os.path.join(args.dst, fn)
            if fn not in ("config.json", "model.safetensors.index.json") and not os.path.exists(dst) and os.path.exists(src):
                shutil.copy(src, dst)
                print(f"[fill] copied {fn} from {args.fill_from}")

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path))
    eos = gc.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if 106 not in eos:
        eos = sorted(set(eos + [1, 106]))
        gc["eos_token_id"] = eos
        print(f"[fix] eos_token_id did not contain 106; set to {eos}")
    if args.greedy:
        gc["do_sample"] = False
        gc.pop("top_k", None)
        gc.pop("top_p", None)
        gc["temperature"] = 0.0
    json.dump(gc, open(gc_path, "w"), indent=2)

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0]
    assert "gemma" in arch.lower(), arch
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(args.dst, f))]

    print(json.dumps({
        "dst": args.dst,
        "architectures": cfg["architectures"],
        "generation_config": gc,
        "missing_files": missing,
        "weight_shards": sorted(f for f in os.listdir(args.dst) if f.endswith(".safetensors")),
    }, indent=2))
    assert not missing, missing

    if args.verify_load:
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        tok = AutoTokenizer.from_pretrained(args.dst)
        m = Gemma3ForConditionalGeneration.from_pretrained(
            args.dst, dtype=torch.bfloat16, device_map="cpu"
        )
        n = sum(p.numel() for p in m.parameters())
        print(f"[verify] loaded on CPU: {n/1e9:.2f}B params, vocab {len(tok)}")


if __name__ == "__main__":
    main()
