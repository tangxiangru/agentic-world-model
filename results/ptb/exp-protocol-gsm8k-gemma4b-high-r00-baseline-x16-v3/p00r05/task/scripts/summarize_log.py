#!/usr/bin/env python3
"""Summarize an inspect_ai gsm8k eval log: accuracy + termination/format diagnostics."""
from __future__ import annotations

import json
import sys


def main(path: str, dump: str | None = None) -> None:
    d = json.load(open(path))
    res = d.get("results") or {}
    scores = res.get("scores") or []
    acc = None
    for s in scores:
        for k, v in (s.get("metrics") or {}).items():
            if k == "accuracy":
                acc = v["value"]
    samples = d.get("samples") or []
    n = len(samples)
    no_marker = trunc = 0
    lens = []
    wrong = []
    for s in samples:
        out = ""
        try:
            out = s["output"]["choices"][0]["message"]["content"]
            if isinstance(out, list):
                out = "".join(c.get("text", "") for c in out)
        except Exception:
            pass
        stop = None
        try:
            stop = s["output"]["choices"][0]["stop_reason"]
        except Exception:
            pass
        lens.append(len(out))
        if "ANSWER:" not in out:
            no_marker += 1
        if stop not in ("stop", None):
            trunc += 1
        sc = (s.get("scores") or {}).get("match") or {}
        if sc.get("value") != "C":
            wrong.append(
                {
                    "id": s.get("id"),
                    "question": s.get("input") if isinstance(s.get("input"), str) else None,
                    "target": s.get("target"),
                    "answer": sc.get("answer"),
                    "tail": out[-400:],
                }
            )
    print(json.dumps({
        "path": path, "accuracy": acc, "n": n,
        "no_answer_marker": no_marker, "not_stopped": trunc,
        "mean_out_chars": round(sum(lens) / max(1, len(lens))),
        "max_out_chars": max(lens) if lens else 0,
        "n_wrong": len(wrong),
    }, indent=2))
    if dump:
        with open(dump, "w") as f:
            for w in wrong:
                f.write(json.dumps(w) + "\n")
        print("wrote", dump)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
