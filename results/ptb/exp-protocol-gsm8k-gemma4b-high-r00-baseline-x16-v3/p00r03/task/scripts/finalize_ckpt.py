"""Make a Trainer checkpoint directory loadable by vLLM (and by the grader).

Trainer's intermediate checkpoint-N/ dirs carry weights + config but no
tokenizer/processor files, which vLLM needs. Copy the missing ones from the
base snapshot, then load the result once on CPU to prove it works.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_SNAPSHOT  # noqa: E402

NEEDED = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
]


def finalize(ckpt, src=BASE_SNAPSHOT, verify=False):
    copied = []
    for fn in NEEDED:
        d = os.path.join(ckpt, fn)
        s = os.path.join(src, fn)
        if not os.path.exists(d) and os.path.exists(s):
            shutil.copyfile(s, d)
            copied.append(fn)
    print(f"{ckpt}: copied {copied}")
    missing = [f for f in ["config.json", "tokenizer.json", "tokenizer_config.json"]
               if not os.path.exists(os.path.join(ckpt, f))]
    if missing:
        raise SystemExit(f"still missing: {missing}")
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    print("architectures:", cfg.get("architectures"))
    if verify:
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        tok = AutoTokenizer.from_pretrained(ckpt)
        m = Gemma3ForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        n = sum(p.numel() for p in m.parameters())
        print(f"CPU load ok: {n/1e9:.2f}B params, vocab {len(tok)}, eot id {tok.convert_tokens_to_ids('<end_of_turn>')}")
        del m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    for c in a.ckpts:
        finalize(c, verify=a.verify)
