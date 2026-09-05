#!/usr/bin/env python3
"""Summarise an inspect_ai eval log: accuracy, format-failure share, samples."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NUM = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def last_number(text: str):
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w2 = w.strip("$%.,:;()[]{}*").replace(",", "")
        if w2 and w2.replace(".", "").replace("-", "").isnumeric():
            return w2
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--dump", default=None, help="write per-sample jsonl here")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    p = Path(args.log_dir)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    files = [f for f in files if f.stat().st_size > 0]
    log = json.loads(files[-1].read_text())
    print("log:", files[-1])
    res = log.get("results") or {}
    for s in res.get("scores", []):
        for k, v in s.get("metrics", {}).items():
            print(f"  {k}: {v['value']}")
    print("  total_samples:", res.get("total_samples"))

    samples = log.get("samples") or []
    out, n_fmt, n_correct, n_trunc = [], 0, 0, 0
    for s in samples:
        comp = ""
        msgs = s.get("messages") or []
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, list):
                    comp = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                else:
                    comp = c or ""
                break
        score = None
        for v in (s.get("scores") or {}).values():
            score = v.get("value")
        ok = score == "C"
        n_correct += ok
        has_marker = "ANSWER:" in comp
        if not has_marker:
            n_fmt += 1
        sr = (s.get("output") or {}).get("stop_reason") or ""
        if sr not in ("stop", "", None):
            n_trunc += 1
        out.append({
            "id": s.get("id"), "target": s.get("target"), "correct": ok,
            "has_answer_marker": has_marker, "stop_reason": sr,
            "n_chars": len(comp),
            "input": (s.get("input") if isinstance(s.get("input"), str) else "")[:2000],
            "completion": comp,
        })
    n = len(samples)
    if n:
        print(f"  parsed n={n} correct={n_correct} ({n_correct/n:.4f})")
        print(f"  no 'ANSWER:' in completion: {n_fmt} ({n_fmt/n:.3f})")
        print(f"  non-stop finish (length cap etc): {n_trunc} ({n_trunc/n:.3f})")
        wrong_fmt = sum(1 for o in out if not o["correct"] and not o["has_answer_marker"])
        print(f"  wrong AND no marker: {wrong_fmt}")
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w") as fh:
            for o in out:
                fh.write(json.dumps(o) + "\n")
        print("  dumped ->", args.dump)
    for o in out[: args.show]:
        print("=" * 70)
        print("target", o["target"], "correct", o["correct"], "stop", o["stop_reason"])
        print(o["completion"][:1500])


if __name__ == "__main__":
    main()
