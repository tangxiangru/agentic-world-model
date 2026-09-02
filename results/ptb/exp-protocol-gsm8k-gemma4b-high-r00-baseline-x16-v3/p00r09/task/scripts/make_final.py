#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ and verify the grader can load it.

Checks, in order:
  * every file vLLM needs is present (config, weights, tokenizer, generation_config)
  * evaluate.py's model_type() resolves to 'gemma' from config.json
  * generation_config eos_token_id contains 106 (<end_of_turn>), the token the
    training targets end with
  * transformers can instantiate the model from the directory (meta device, fast)
"""
import argparse
import json
import os
import shutil
import sys

NEEDED = ["config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "special_tokens_map.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--base", required=True, help="base snapshot, for missing aux files")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst, symlinks=False)
    # drop trainer state that the grader does not need
    for junk in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth",
                 "training_args.bin"):
        p = os.path.join(args.dst, junk)
        if os.path.exists(p):
            os.remove(p)
    for f in NEEDED + ["preprocessor_config.json", "processor_config.json",
                       "tokenizer.model", "added_tokens.json"]:
        d = os.path.join(args.dst, f)
        s = os.path.join(args.base, f)
        if not os.path.exists(d) and os.path.exists(s):
            shutil.copy(s, d)

    problems = []
    for f in NEEDED:
        if not os.path.exists(os.path.join(args.dst, f)):
            problems.append(f"missing {f}")
    if not any(x.endswith(".safetensors") for x in os.listdir(args.dst)):
        problems.append("no safetensors weights")

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0].lower()
    if "gemma" not in arch:
        problems.append(f"evaluate.py model_type() would not resolve: {arch}")

    gc = json.load(open(os.path.join(args.dst, "generation_config.json")))
    eos = gc.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if 106 not in eos:
        problems.append(f"generation_config eos_token_id={eos} lacks 106 <end_of_turn>")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.dst)
    assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
    AutoConfig.from_pretrained(args.dst)
    import torch
    m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16,
                                             device_map="cpu")
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded {args.dst} on CPU: {arch}, {n/1e9:.2f}B params, eos={eos}")
    du = sum(os.path.getsize(os.path.join(args.dst, f)) for f in os.listdir(args.dst))
    print(f"size {du/1e9:.1f} GB, files: {sorted(os.listdir(args.dst))}")
    if problems:
        print("PROBLEMS:", problems)
        sys.exit(1)
    print("final_model OK")


if __name__ == "__main__":
    main()
