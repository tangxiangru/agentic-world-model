#!/usr/bin/env python3
"""Token-length profile of a training jsonl, rendered exactly as the trainer will.

Prints p50/p95/p99/max of the full row and of the completion, and the share of
rows that would truncate at a candidate max_seq_len. This is the seq_len_truncation
pitfall check done for real rather than by the chars/4 estimate.
"""
import argparse
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_sft import render  # noqa: E402

SNAP = os.environ["PTB_BASE_MODEL_SNAPSHOT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    tot, comp = [], []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            r = json.loads(line)
            pre, full = render(r["prompt"], r["completion"])
            np_ = len(tok(pre, add_special_tokens=False)["input_ids"])
            nf = len(tok(full, add_special_tokens=False)["input_ids"])
            tot.append(nf)
            comp.append(nf - np_)
    tot, comp = np.array(tot), np.array(comp)
    for name, a in (("total", tot), ("completion", comp)):
        print(
            f"{name}: p50={np.percentile(a,50):.0f} p95={np.percentile(a,95):.0f} "
            f"p99={np.percentile(a,99):.0f} max={a.max()} mean={a.mean():.0f}"
        )
    for L in (1024, 1280, 1536, 2048):
        print(f"  truncated at {L}: {(tot > L).mean()*100:.3f}%")
    print(f"chosen max_seq_len={args.max_seq_len}: {(tot > args.max_seq_len).mean()*100:.3f}% truncate")
    print(f"total tokens over {len(tot)} rows: {tot.sum()/1e6:.1f}M (mean {tot.mean():.0f})")


if __name__ == "__main__":
    main()
