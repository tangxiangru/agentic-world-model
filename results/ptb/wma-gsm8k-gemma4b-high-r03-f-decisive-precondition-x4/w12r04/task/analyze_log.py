#!/usr/bin/env python3
"""Read an inspect-ai json eval log and report the diagnostics the cards promise.

  * accuracy as the harness scored it
  * clean-termination rate: last non-empty line is exactly `ANSWER: <number>`
  * share of completions that hit the max-tokens cap
  * per-item records, written to a jsonl for watch-set work
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

MARKER_RE = re.compile(r"^ANSWER:\s*-?[\d.,]+\s*$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, "*.json")), key=os.path.getmtime)
    files = [f for f in files if not f.endswith("summary.json")]
    assert files, f"no json log under {args.log_dir}"
    log = json.load(open(files[-1]))
    samples = log["samples"]

    recs, clean, capped, correct = [], 0, 0, 0
    for s in samples:
        comp = ""
        for m in reversed(s.get("messages", [])):
            if m.get("role") == "assistant":
                c = m.get("content")
                comp = c if isinstance(c, str) else "".join(
                    p.get("text", "") for p in c if isinstance(p, dict)
                )
                break
        score = list(s["scores"].values())[0]
        ok = score["value"] == "C"
        correct += ok
        lines = [ln for ln in comp.strip().split("\n") if ln.strip()]
        is_clean = bool(lines) and bool(MARKER_RE.match(lines[-1].strip()))
        clean += is_clean
        stop = (s.get("output") or {}).get("stop_reason")
        hit_cap = stop == "max_tokens"
        capped += hit_cap
        recs.append({
            "id": s["id"], "question": s["input"] if isinstance(s["input"], str) else str(s["input"]),
            "gold": s["target"], "answer": score.get("answer"), "correct": ok,
            "clean": is_clean, "capped": hit_cap, "n_chars": len(comp),
            "completion": comp,
        })

    n = len(samples)
    print(f"log            : {files[-1]}")
    print(f"n              : {n}")
    print(f"accuracy       : {correct / n:.4f}  ({correct}/{n})")
    print(f"clean-term rate: {clean / n:.4f}")
    print(f"hit max_tokens : {capped / n:.4f}")
    print(f"mean chars     : {sum(r['n_chars'] for r in recs) / n:.0f}")
    if args.out:
        with open(args.out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
