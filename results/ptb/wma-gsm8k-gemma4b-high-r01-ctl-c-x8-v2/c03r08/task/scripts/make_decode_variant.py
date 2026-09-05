#!/usr/bin/env python3
"""Clone a checkpoint dir by symlink, overriding only generation_config.json.

vLLM logs "Default sampling parameters have been overridden by the model's
Hugging Face generation config", so the checkpoint's generation_config.json - not
evaluate.py - decides how the grader samples. The stock gemma-3 config carries
do_sample/top_k=64/top_p=0.95 and no temperature, i.e. the benchmark is graded
under temperature-1.0 sampling. This makes a variant that decodes greedily,
without touching the weights or evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)  # 0 = disabled in vLLM
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name == "generation_config.json":
            continue
        link = dst / p.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(p.resolve(), link)

    gc = json.loads((src / "generation_config.json").read_text())
    gc["do_sample"] = args.temperature > 0
    gc["temperature"] = args.temperature
    gc["top_p"] = args.top_p
    gc["top_k"] = args.top_k
    (dst / "generation_config.json").write_text(json.dumps(gc, indent=2))
    print(json.dumps(gc, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
