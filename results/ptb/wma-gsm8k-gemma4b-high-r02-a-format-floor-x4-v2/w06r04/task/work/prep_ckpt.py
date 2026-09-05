"""Make a Trainer checkpoint-N/ directory loadable by vLLM.

Trainer writes weights + config + generation_config only; the tokenizer,
processor and preprocessor files stay in the parent snapshot. vLLM needs them,
so copy them in. Verified against work/smoke3/checkpoint-3.
"""
from __future__ import annotations

import argparse
import os
import shutil

FILES = [
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "preprocessor_config.json", "processor_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parent", required=True)
    args = ap.parse_args()
    copied = []
    for f in FILES:
        src = os.path.join(args.parent, f)
        dst = os.path.join(args.ckpt, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
            copied.append(f)
    gc = os.path.join(args.ckpt, "generation_config.json")
    if not os.path.exists(gc):
        shutil.copyfile(os.path.join(args.parent, "generation_config.json"), gc)
        copied.append("generation_config.json")
    print("copied:", copied or "(nothing missing)")


if __name__ == "__main__":
    main()
