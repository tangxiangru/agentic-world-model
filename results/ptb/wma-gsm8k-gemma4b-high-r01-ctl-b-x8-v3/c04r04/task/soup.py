#!/usr/bin/env python3
"""Uniform weight average (model soup) of checkpoints that share an init lineage."""
import argparse, json, os, shutil

import torch
from transformers import AutoModelForCausalLM

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
AUX = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
       "added_tokens.json", "preprocessor_config.json", "processor_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.src[0], dtype=torch.float32)
    acc = {k: v.clone().float() for k, v in model.state_dict().items()}
    for extra in args.src[1:]:
        m2 = AutoModelForCausalLM.from_pretrained(extra, dtype=torch.float32)
        sd = m2.state_dict()
        assert set(sd) == set(acc), "state dicts differ"
        for k in acc:
            acc[k] += sd[k].float()
        del m2
    n = len(args.src)
    for k in acc:
        acc[k] /= n
    model.load_state_dict({k: v for k, v in acc.items()})
    model = model.to(torch.bfloat16)
    os.makedirs(args.dst, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)
    for f in AUX:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.dst, f))
    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    gc["temperature"] = args.temperature
    if args.temperature == 0.0:
        gc["do_sample"] = False
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print("souped", n, "checkpoints ->", args.dst)


if __name__ == "__main__":
    main()
