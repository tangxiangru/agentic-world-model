#!/usr/bin/env python3
"""Read an inspect json eval log and report the diagnostics --json-output-file omits.

  accuracy, format compliance ('ANSWER:' present), termination (stop_reason),
  completion-length distribution, and a few failing completions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--show", type=int, default=0, help="print N incorrect completions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, "*.json")), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no json log in {args.log_dir}")
    log = json.load(open(files[-1]))
    samples = log.get("samples") or []

    n = len(samples)
    correct = fmt = 0
    stops = Counter()
    lens = []
    wrong = []
    for s in samples:
        comp = ""
        try:
            comp = s["output"]["choices"][0]["message"]["content"]
            if not isinstance(comp, str):
                comp = "".join(c.get("text", "") for c in comp)
        except Exception:
            pass
        try:
            stops[s["output"]["choices"][0].get("stop_reason", "?")] += 1
        except Exception:
            stops["?"] += 1
        try:
            lens.append(s["output"]["usage"]["output_tokens"])
        except Exception:
            pass
        sc = list((s.get("scores") or {}).values())
        ok = bool(sc) and sc[0].get("value") == "C"
        correct += ok
        fmt += "ANSWER:" in comp
        if not ok:
            wrong.append({"id": s.get("id"), "target": s.get("target"), "completion": comp})

    lens.sort()
    rep = {
        "log": files[-1],
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "answer_marker_share": round(fmt / n, 4) if n else None,
        "stop_reasons": dict(stops),
        "out_tokens_p50": lens[len(lens) // 2] if lens else None,
        "out_tokens_p95": lens[int(len(lens) * 0.95)] if lens else None,
        "out_tokens_max": lens[-1] if lens else None,
    }
    print(json.dumps(rep, indent=2))
    if args.out:
        json.dump({**rep, "wrong": wrong}, open(args.out, "w"), indent=2)
    for w in wrong[: args.show]:
        print("=" * 70)
        print("id", w["id"], "gold", w["target"])
        print(w["completion"][:1500])


if __name__ == "__main__":
    main()
