#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k eval log: score, format diagnostic, failures."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

WELL_FORMED = re.compile(r"ANSWER:\s*-?[\d,]+(\.\d+)?\s*$")


def latest_log(pattern: str = "logs/*gsm8k*.json") -> str:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--failures-out", default=None)
    args = ap.parse_args()

    path = args.log or latest_log()
    d = json.load(open(path))
    samples = d["samples"]
    n = len(samples)
    correct, well_formed, stops = 0, 0, Counter()
    failures = []
    lens = []
    for s in samples:
        ch = s["output"]["choices"][0]
        text = ch["message"]["content"].strip()
        ok = s["scores"]["match"]["value"] == "C"
        correct += ok
        stops[ch.get("stop_reason")] += 1
        wf = bool(WELL_FORMED.search(text)) and ch.get("stop_reason") == "stop"
        well_formed += wf
        lens.append(len(text))
        if not ok:
            failures.append({
                "id": s["id"], "target": s["target"],
                "well_formed": wf, "stop_reason": ch.get("stop_reason"),
                "question": s["input"] if isinstance(s["input"], str) else None,
                "output_tail": text[-600:],
            })
    lens.sort()
    summary = {
        "log": path,
        "n": n,
        "accuracy": correct / n,
        "well_formed_and_stopped": well_formed / n,
        "stop_reasons": dict(stops),
        "output_chars_p50": lens[n // 2],
        "output_chars_p95": lens[int(n * 0.95)],
        "n_failures": n - correct,
        "failures_not_well_formed": sum(1 for f in failures if not f["well_formed"]),
    }
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(args.out, "w"), indent=2)
    if args.failures_out:
        json.dump(failures, open(args.failures_out, "w"), indent=2)


if __name__ == "__main__":
    main()
