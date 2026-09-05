"""Copy a checkpoint into final_model/ and verify the grader can actually load it.

Guards the `final_model_not_loadable` pitfall: the grader loads final_model/ with vLLM
from a fresh process, so the directory must carry weights, tokenizer, processor configs
and a generation_config with eos_token_id [1, 106].
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

REQUIRED = [
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
    ap.add_argument("--check-load", action="store_true", help="load on CPU with transformers")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for name in sorted(os.listdir(args.src)):
        s, d = os.path.join(args.src, name), os.path.join(args.dst, name)
        if os.path.isdir(s):
            continue
        shutil.copy(s, d)  # real copy: final_model must survive ckpts/ being cleaned up

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(args.dst, f))]
    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    gen = json.load(open(os.path.join(args.dst, "generation_config.json")))
    report = {
        "src": args.src,
        "dst": args.dst,
        "missing_required_files": missing,
        "architectures": cfg.get("architectures"),
        "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
        "eos_token_id": gen.get("eos_token_id"),
        "generation_config": gen,
        "size_gb": round(sum(
            os.path.getsize(os.path.join(args.dst, f)) for f in os.listdir(args.dst)
        ) / 1e9, 2),
    }
    if args.check_load:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.dst)
        m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16)
        report["cpu_load"] = "ok"
        report["n_params_b"] = round(sum(p.numel() for p in m.parameters()) / 1e9, 3)
        report["tokenizer_end_of_turn_id"] = tok("<end_of_turn>", add_special_tokens=False)["input_ids"]
        del m
    print(json.dumps(report, indent=2))
    assert not missing, f"missing required files: {missing}"
    assert 106 in (gen.get("eos_token_id") or []), "eos_token_id must contain 106 (<end_of_turn>)"


if __name__ == "__main__":
    main()
