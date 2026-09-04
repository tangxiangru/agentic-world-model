#!/usr/bin/env python3
"""Score several checkpoints of one run on the held-out GSM8K-TRAIN probe.

Cheap stand-in for the 150-item harness read used only to pick which
checkpoint(s) are worth a real evaluate.py run (checkpoint selection).
Each checkpoint is first passed through finalize_model.py, because the
intermediate trainer dirs carry no tokenizer/processor and vLLM cannot load
them raw, and because the raw dirs would otherwise be scored with the base
snapshot's sampling generation_config instead of greedy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--keep-eval-dirs", action="store_true")
    args = ap.parse_args()

    results = {}
    for ck in args.ckpts:
        name = os.path.basename(ck.rstrip("/"))
        dst = ck.rstrip("/") + "_eval"
        if not os.path.exists(os.path.join(dst, "tokenizer.json")):
            subprocess.run(
                [sys.executable, "finalize_model.py", "--src", ck, "--dst", dst],
                check=True,
            )
        out = f"analysis/{args.tag}_{name}_probe.json"
        r = subprocess.run(
            [
                sys.executable, "probe_eval.py", "--model", dst,
                "--limit", str(args.limit), "--out", out,
            ],
            capture_output=True, text=True,
        )
        line = [l for l in r.stdout.splitlines() if "probe accuracy" in l]
        print(name, line[-1] if line else r.stdout[-500:] + r.stderr[-2000:])
        if os.path.exists(out):
            results[name] = json.load(open(out))["accuracy"]
        if not args.keep_eval_dirs and name != "final":
            pass  # keep them; disk is cheap and re-running finalize costs a copy
    print(json.dumps(results, indent=2))
    with open(f"analysis/{args.tag}_probe_sweep.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
