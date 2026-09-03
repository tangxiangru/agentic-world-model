#!/usr/bin/env python3
"""Read the newest inspect eval log and report accuracy + the format diagnostic."""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics
import sys


def main() -> None:
    log = sys.argv[1] if len(sys.argv) > 1 else max(
        glob.glob("logs/*_gsm8k_*.json"), key=os.path.getmtime
    )
    out = sys.argv[2] if len(sys.argv) > 2 else None
    d = json.load(open(log))
    sm = d["samples"]
    ok = endsans = 0
    sr = collections.Counter()
    lens = []
    for s in sm:
        ch = s["output"]["choices"][0]
        c = ch["message"]["content"]
        sr[str(ch.get("stop_reason"))] += 1
        u = s["output"].get("usage") or {}
        lens.append(u.get("output_tokens") or 0)
        lines = [x for x in c.strip().split("\n") if x.strip()]
        if lines and re.match(r"^ANSWER:\s*-?[\d,\.]+\s*$", lines[-1].strip()):
            endsans += 1
        if s["scores"]["match"]["value"] == "C":
            ok += 1
    res = {
        "log": log,
        "model": d["eval"]["model"],
        "n": len(sm),
        "accuracy": ok / len(sm),
        "ends_with_answer_line": endsans / len(sm),
        "stop_reasons": dict(sr),
        "output_tokens_p50": statistics.median(lens),
        "output_tokens_max": max(lens),
        "max_connections": d["eval"].get("config", {}).get("max_connections"),
    }
    print(json.dumps(res, indent=2))
    if out:
        json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
