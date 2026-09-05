#!/usr/bin/env python3
"""Read an inspect json eval log and report the diagnostics the cards ask for."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

ANSWER_LINE = re.compile(r"^ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\.?$")


def latest_log(pattern: str = "logs/*_gsm8k_*.json") -> str:
    return sorted(glob.glob(pattern), key=os.path.getmtime)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=None, help="write the per-item summary here")
    ap.add_argument("--watch-out", default=None, help="write failing items as jsonl")
    ap.add_argument("--n-examples", type=int, default=5)
    args = ap.parse_args()

    path = args.log or latest_log()
    d = json.load(open(path))
    samples = d["samples"]

    stop = Counter()
    fmt_ok = 0
    correct = 0
    items = []
    for s in samples:
        ch = s["output"]["choices"][0]
        out = ch["message"]["content"]
        sr = ch.get("stop_reason")
        stop[sr] += 1
        lines = [ln.strip() for ln in out.strip().split("\n") if ln.strip()]
        ok = bool(lines) and bool(ANSWER_LINE.match(lines[-1]))
        fmt_ok += ok
        sc = s.get("scores", {}).get("match", {})
        val = sc.get("value")
        c = val == "C" or val == 1 or val is True
        correct += bool(c)
        items.append({
            "id": s["id"], "target": s["target"], "correct": bool(c),
            "fmt_ok": ok, "stop_reason": sr, "n_chars": len(out),
            "question": s["input"] if isinstance(s["input"], str) else None,
            "answer_extracted": sc.get("answer"),
        })

    n = len(samples)
    print(f"log: {path}")
    print(f"n={n} accuracy={correct/n:.4f} format_compliant={fmt_ok/n:.4f}")
    print("stop_reason:", dict(stop))
    lens = sorted(i["n_chars"] for i in items)
    print(f"output chars p50={lens[n//2]} p95={lens[int(n*.95)]} max={lens[-1]}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"log": path, "n": n, "accuracy": correct / n,
                       "format_compliant": fmt_ok / n, "stop_reason": dict(stop),
                       "items": items}, f, indent=1)
        print("wrote", args.out)
    if args.watch_out:
        with open(args.watch_out, "w") as f:
            for i in items:
                if not i["correct"]:
                    f.write(json.dumps({"id": i["id"], "question": i["question"],
                                        "gold": i["target"]}) + "\n")
        print("wrote", args.watch_out, sum(1 for i in items if not i["correct"]), "failures")


if __name__ == "__main__":
    main()
