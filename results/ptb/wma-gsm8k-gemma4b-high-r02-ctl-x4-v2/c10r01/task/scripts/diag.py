#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k eval log: accuracy, clean-termination share,
stop reasons, and the failing items."""
from __future__ import annotations

import collections
import glob
import json
import os
import sys


def analyse(path: str) -> dict:
    d = json.load(open(path))
    s = d["samples"]
    recs, clean = [], 0
    for x in s:
        c = x["output"]["choices"][0]["message"]["content"]
        sr = x["output"]["choices"][0].get("stop_reason")
        lines = [l for l in c.strip().split("\n") if l.strip()]
        last_is_answer = bool(lines) and lines[-1].strip().startswith("ANSWER:")
        ok = (sr == "stop") and last_is_answer
        clean += ok
        recs.append(
            {
                "id": x["id"],
                "target": x["target"],
                "score": x["scores"]["match"]["value"],
                "graded_answer": x["scores"]["match"]["answer"],
                "stop_reason": sr,
                "clean": ok,
                "n_chars": len(c),
                "question": x["input"] if isinstance(x["input"], str) else None,
                "completion": c,
            }
        )
    acc = sum(r["score"] == "C" for r in recs) / len(recs)
    return {
        "log": path,
        "n": len(recs),
        "accuracy": acc,
        "clean_termination": clean,
        "clean_frac": clean / len(recs),
        "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in recs)),
        "median_chars": sorted(r["n_chars"] for r in recs)[len(recs) // 2],
        "per_sample": recs,
    }


if __name__ == "__main__":
    log = sys.argv[1]
    if os.path.isdir(log):
        log = max(glob.glob(os.path.join(log, "*_gsm8k_*.json")), key=os.path.getmtime)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    r = analyse(log)
    print(
        json.dumps(
            {k: v for k, v in r.items() if k != "per_sample"}, indent=1
        )
    )
    if out:
        json.dump(r, open(out, "w"), indent=1)
        print("wrote", out)
