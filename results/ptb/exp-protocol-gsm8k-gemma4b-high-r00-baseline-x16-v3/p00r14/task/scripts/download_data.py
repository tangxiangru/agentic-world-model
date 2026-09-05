"""Pull the candidate training corpora into the HF cache and report their shape.

Nothing here is filtered or formatted; build_sft_data.py does that. This script
exists only so the (slow) network fetch happens once, in the background.
"""
import sys
from collections import Counter

from datasets import load_dataset


def probe(name, **kw):
    print(f"=== {name} {kw} ===", flush=True)
    try:
        d = load_dataset(name, **kw)
        print(d, flush=True)
        row = d[0] if not hasattr(d, "keys") else d[list(d.keys())[0]][0]
        for k, v in row.items():
            print(f"  {k}: {str(v)[:400]!r}", flush=True)
        if "problem_source" in getattr(d, "column_names", []):
            print(Counter(d["problem_source"]), flush=True)
        if "type" in getattr(d, "column_names", []):
            print(Counter(d["type"]), flush=True)
    except Exception as e:  # noqa: BLE001 - a missing hub dataset must not stop the rest
        print(f"  ERR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "omi"):
        probe("nvidia/OpenMathInstruct-2", split="train_1M")
    if which in ("all", "meta"):
        probe("meta-math/MetaMathQA", split="train")
    if which in ("all", "orca"):
        probe("microsoft/orca-math-word-problems-200k", split="train")
    print("DONE", flush=True)
