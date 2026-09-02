#!/usr/bin/env python3
"""Make a decode-config variant of a checkpoint without copying the weights.

Large files are symlinked, small json/txt files are copied, and
generation_config.json is rewritten. vLLM adopts the checkpoint's
generation_config as its default sampling params (it logs
"Default sampling parameters have been overridden by the model's Hugging Face
generation config"), so this is how the decoding used at grading time is set.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

BIG = {".safetensors", ".bin", ".pt", ".model"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--greedy", type=int, default=1)
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for f in src.iterdir():
        if f.is_dir():
            continue
        if f.suffix in BIG:
            os.symlink(f, dst / f.name)
        else:
            shutil.copy2(f, dst / f.name)

    gc_path = dst / "generation_config.json"
    gc = json.loads(gc_path.read_text()) if gc_path.exists() else {}
    if args.greedy:
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc["top_p"] = 1.0
        gc["top_k"] = -1
        for k in ("min_p", "typical_p"):
            gc.pop(k, None)
    gc_path.write_text(json.dumps(gc, indent=2) + "\n")
    print(json.dumps(gc, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
