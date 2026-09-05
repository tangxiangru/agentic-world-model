#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ behind a regression guard, set the decode
arm, and verify the result loads the way the grader will load it.

The grader runs `python evaluate.py --model-path final_model` from a fresh
process: it needs config.json (architectures -> gemma), the weights, the
tokenizer, the processor configs, and a generation_config.json whose
eos_token_id contains 106 (<end_of_turn>).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

TASK = "/home/ben/task"
GUARD = os.path.join(TASK, "final_model", "PACKAGED.json")

REQUIRED = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--score", type=float, required=True, help="measured dev-150 accuracy of --ckpt")
    ap.add_argument("--protocol", required=True, help="one line describing how --score was measured")
    ap.add_argument("--decode", choices=["greedy", "sampling", "keep"], default="greedy")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    prev = None
    if os.path.exists(GUARD):
        prev = json.load(open(GUARD))
    if prev and not args.force and args.score <= prev.get("score", -1):
        print(
            f"REFUSING: final_model already holds {prev['ckpt']} at {prev['score']} "
            f"(protocol: {prev['protocol']}); candidate scores {args.score}."
        )
        sys.exit(2)

    dst = os.path.join(TASK, "final_model")
    tmp = os.path.join(TASK, "final_model.new")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(args.ckpt, tmp, symlinks=False)
    for junk in ("trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        p = os.path.join(tmp, junk)
        if os.path.exists(p):
            os.remove(p)

    if args.decode != "keep":
        subprocess.check_call(
            [sys.executable, os.path.join(TASK, "scripts", "set_decode.py"), "--ckpt", tmp, "--mode", args.decode]
        )
        gcp = os.path.join(tmp, "generation_config.json.orig")
        if os.path.exists(gcp):
            os.remove(gcp)

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(tmp, f))]
    if missing:
        print(f"REFUSING: {missing} missing from {args.ckpt}")
        sys.exit(3)

    gc = json.load(open(os.path.join(tmp, "generation_config.json")))
    eos = gc["eos_token_id"]
    assert (106 in eos) if isinstance(eos, list) else eos == 106, gc

    # the grader's own model-type detection must resolve to the gemma template
    cfg = json.load(open(os.path.join(tmp, "config.json")))
    assert "gemma" in cfg["architectures"][0].lower(), cfg["architectures"]

    # load once, from a fresh process, the way transformers/vLLM will
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            (
                "import torch, transformers;"
                f"m=transformers.AutoModelForCausalLM.from_pretrained('{tmp}', dtype=torch.bfloat16);"
                f"t=transformers.AutoTokenizer.from_pretrained('{tmp}');"
                "print('[load] ok', type(m).__name__, sum(p.numel() for p in m.parameters())/1e9, 'B',"
                " 'eos', t.convert_tokens_to_ids('<end_of_turn>'))"
            ),
        ]
    )

    shutil.rmtree(dst, ignore_errors=True)
    os.rename(tmp, dst)
    json.dump(
        {"ckpt": args.ckpt, "score": args.score, "protocol": args.protocol, "decode": args.decode},
        open(GUARD, "w"),
        indent=2,
    )
    print(f"[package] final_model <- {args.ckpt} (score {args.score}, decode {args.decode})")
    print(json.dumps(gc, indent=2))


if __name__ == "__main__":
    main()
