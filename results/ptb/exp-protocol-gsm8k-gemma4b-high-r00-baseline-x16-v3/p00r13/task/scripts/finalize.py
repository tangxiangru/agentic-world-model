#!/usr/bin/env python3
"""Assemble final_model/ from a checkpoint and prove the grader can load it.

Guards the `final_model_not_loadable` pitfall: copies weights + tokenizer +
processor, writes the chosen decode config, then loads the directory from a
fresh process on CPU with transformers and renders one prompt through the
grader's own template.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent.parent
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

KEEP_FROM_SNAP = ["preprocessor_config.json", "processor_config.json",
                  "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=str(TASK / "final_model"))
    ap.add_argument("--mode", default="greedy")
    a = ap.parse_args()

    src, dst = Path(a.ckpt), Path(a.out)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in ("optimizer.pt", "scheduler.pt", "rng_state.pth",
                                          "trainer_state.json", "training_args.bin"):
            shutil.copy2(f, dst / f.name)
    for name in KEEP_FROM_SNAP:
        p = Path(SNAP) / name
        if p.exists() and not (dst / name).exists():
            shutil.copy2(p, dst / name)
    subprocess.run([sys.executable, str(TASK / "scripts" / "set_gen_config.py"),
                    str(dst), "--mode", a.mode], check=True)

    print("files:", sorted(p.name for p in dst.iterdir()))
    # fresh-process CPU load, exactly what the grader's loader has to do
    code = f'''
import json, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
d = {str(dst)!r}
cfg = AutoConfig.from_pretrained(d); print("arch:", cfg.architectures)
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16)
print("params:", sum(p.numel() for p in m.parameters())/1e9, "B")
print("gen cfg:", json.load(open(d + "/generation_config.json")))
tpl = open({str(TASK / "templates" / "gemma3.jinja")!r}).read()
s = tok.apply_chat_template([{{"role":"user","content":"hi"}}], chat_template=tpl,
                            tokenize=False, add_generation_prompt=True)
print("template render ok:", repr(s))
'''
    subprocess.run([sys.executable, "-c", code], check=True)
    print("final_model is loadable")


if __name__ == "__main__":
    main()
