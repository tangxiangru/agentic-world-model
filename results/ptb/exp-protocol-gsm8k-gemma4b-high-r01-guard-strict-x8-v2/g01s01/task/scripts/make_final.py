"""Assemble final_model/ from a checkpoint: weights + tokenizer + decode config,
then load it once from a fresh process to prove the grader can.

Guards pitfall `final_model_not_loadable`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "model.safetensors.index.json"]


def write_generation_config(dst, greedy):
    p = os.path.join(dst, "generation_config.json")
    cfg = json.load(open(p)) if os.path.exists(p) else {}
    cfg["bos_token_id"] = 2
    cfg["eos_token_id"] = [1, 106]
    cfg["pad_token_id"] = 0
    cfg["cache_implementation"] = "hybrid"
    if greedy:
        # vLLM reads temperature/top_p/top_k from here (ModelConfig.get_diff_sampling_param)
        # and the harness never overrides them, so this is where decoding is chosen.
        cfg["do_sample"] = False
        cfg["temperature"] = 0.0
        cfg.pop("top_k", None)
        cfg.pop("top_p", None)
    else:
        cfg["do_sample"] = True
        cfg["top_k"] = 64
        cfg["top_p"] = 0.95
        cfg.pop("temperature", None)
    json.dump(cfg, open(p, "w"), indent=2)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--sampling", action="store_true", help="keep the base model's sampling config")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        print("removing existing", args.dst)
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)
    for fn in ("preprocessor_config.json", "processor_config.json", "tokenizer.model", "added_tokens.json"):
        s = os.path.join(SNAP, fn)
        d = os.path.join(args.dst, fn)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy(s, d)
    cfg = write_generation_config(args.dst, greedy=not args.sampling)
    print("generation_config:", cfg)

    missing = [f for f in NEEDED if not os.path.exists(os.path.join(args.dst, f))]
    print("files:", sorted(os.listdir(args.dst)))
    if missing:
        print("MISSING:", missing)
        sys.exit(1)

    if not args.no_verify:
        code = (
            "import torch, json;"
            "from transformers import AutoModelForCausalLM, AutoTokenizer;"
            f"m=AutoModelForCausalLM.from_pretrained({args.dst!r}, dtype=torch.bfloat16);"
            f"t=AutoTokenizer.from_pretrained({args.dst!r});"
            "print('loaded', type(m).__name__, sum(p.numel() for p in m.parameters())/1e9, 'B');"
            "print('eos', m.generation_config.eos_token_id, 'temp', m.generation_config.temperature,"
            " 'do_sample', m.generation_config.do_sample);"
            "print('tok ok', t.encode('<end_of_turn>', add_special_tokens=False))"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
            sys.exit(1)
    print("[ok] final_model assembled at", args.dst)


if __name__ == "__main__":
    main()
