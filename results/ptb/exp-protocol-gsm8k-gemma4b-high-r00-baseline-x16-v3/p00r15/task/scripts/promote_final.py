#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ with the greedy generation_config adopted in exp-03.

Weights are copied, not symlinked: final_model/ has to survive on its own when the
grader loads it from a fresh process (pitfall final_model_not_loadable).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        print(f"replacing existing {args.dst}")
        shutil.rmtree(args.dst)
    subprocess.check_call([
        sys.executable, "/home/ben/task/scripts/make_variant.py",
        "--src", os.path.realpath(args.src), "--dst", args.dst, "--greedy", "--copy",
    ])
    # a record of where these weights came from
    json.dump({"source_checkpoint": os.path.realpath(args.src)},
              open(os.path.join(args.dst, "provenance.json"), "w"), indent=2)
    print("promoted", args.src, "->", args.dst)


if __name__ == "__main__":
    main()
