#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ and make it loadable by the grader.

The grader runs `vllm serve final_model` from a fresh process with
templates/gemma3.jinja as the chat template. vLLM 0.11 defaults to
`--generation-config auto`, so whatever sits in final_model/generation_config.json
becomes the default sampling params for the benchmark. eos_token_id must keep
106 (<end_of_turn>) or nothing stops.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--decode", choices=["greedy", "keep"], default="greedy")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)
    for junk in ("optimizer.pt", "scheduler.pt", "trainer_state.json",
                 "training_args.bin", "rng_state.pth"):
        p = os.path.join(args.dst, junk)
        if os.path.exists(p):
            os.remove(p)

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    if args.decode == "greedy":
        gc.update({"do_sample": False, "temperature": 0.0,
                   "top_p": 1.0, "top_k": -1})
    gc.setdefault("bos_token_id", 2)
    gc.setdefault("pad_token_id", 0)
    gc["eos_token_id"] = [1, 106]
    json.dump(gc, open(gc_path, "w"), indent=2)
    print(json.dumps(gc, indent=2))

    need = ["config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(args.dst, f))]
    print("files:", sorted(os.listdir(args.dst)))
    assert not missing, f"missing {missing}"

    if args.verify:
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(args.dst)
        assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
        m = Gemma3ForConditionalGeneration.from_pretrained(
            args.dst, dtype=torch.bfloat16, device_map="cpu")
        n = sum(p.numel() for p in m.parameters())
        print(f"loaded on CPU: {n/1e9:.2f}B params, dtype {m.dtype}")
    print(f"final_model ready at {args.dst}")


if __name__ == "__main__":
    main()
