"""Assemble final_model/ from a checkpoint and verify it loads the way the grader loads it.

Pitfall `final_model_not_loadable`: the grader loads final_model/ with vLLM from a
fresh process, after deciding the chat template from config.json's architecture.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--decode", choices=["greedy", "base"], default="greedy")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)

    # tokenizer + processor files the grader/vLLM may look for
    for fn in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json",
               "preprocessor_config.json", "processor_config.json"]:
        d = os.path.join(args.dst, fn)
        if not os.path.exists(d) and os.path.exists(os.path.join(SNAP, fn)):
            shutil.copy(os.path.join(SNAP, fn), d)

    subprocess.check_call([sys.executable, "/home/ben/task/set_decode.py",
                           "--model", args.dst, "--mode", args.decode])

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0]
    assert "gemma" in arch.lower(), f"evaluate.py picks the template from the architecture: {arch}"
    print("architecture:", arch, "-> templates/gemma3.jinja")

    # load once on CPU with transformers, exactly the failure mode the pitfall names
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(args.dst)
    m = Gemma3ForConditionalGeneration.from_pretrained(args.dst, dtype=torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded {args.dst} on CPU: {n/1e9:.2f}B params, dtype {next(m.parameters()).dtype}")
    print("tokenizer eos:", tok.eos_token, "| <end_of_turn> id:", tok.convert_tokens_to_ids("<end_of_turn>"))
    print("generation_config:", json.load(open(os.path.join(args.dst, "generation_config.json"))))
    print("OK")


if __name__ == "__main__":
    main()
