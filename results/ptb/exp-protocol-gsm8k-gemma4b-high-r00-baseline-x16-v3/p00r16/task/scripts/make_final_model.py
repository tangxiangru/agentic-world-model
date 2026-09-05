"""Assemble final_model/ from a checkpoint and verify the grader can load it.

The grader runs `vllm serve final_model --chat-template templates/gemma3.jinja`
from a fresh process, and vLLM takes its *default sampling parameters* from the
model directory's generation_config.json (it logs "Default sampling parameters
have been overridden by the model's Hugging Face generation config"). So the
decode configuration is part of what a checkpoint ships, and it is set here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)
    for junk in ("training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                 "trainer_state.json"):
        p = os.path.join(args.dst, junk)
        if os.path.exists(p):
            os.remove(p)

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path))
    if args.temperature <= 0:
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    else:
        gc["do_sample"] = True
        gc["temperature"] = args.temperature
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("generation_config.json ->", json.dumps(gc))

    if args.verify:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        print("config:", AutoConfig.from_pretrained(args.dst).architectures)
        tok = AutoTokenizer.from_pretrained(args.dst)
        assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
        m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16,
                                                 device_map="cpu")
        n = sum(p.numel() for p in m.parameters())
        print(f"loaded on CPU with transformers: {n/1e9:.2f}B params")
    print("final_model at", args.dst)
    print(sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
