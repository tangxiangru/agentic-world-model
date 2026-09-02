#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k log: accuracy, stop behaviour, output length.

Prints only aggregate counts and per-item ids/flags -- never benchmark text --
so the output is safe to keep in the experiment record (protocol rule 7).
"""

from __future__ import annotations

import argparse
import glob
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*gsm8k*.json"))[-1]
    d = json.load(open(path))
    samples = d["samples"]
    rows = []
    for s in samples:
        ch = s["output"]["choices"][0]
        text = ch["message"]["content"]
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text)
        rows.append({
            "id": s["id"],
            "correct": s["scores"]["match"]["value"] == "C",
            "stopped": ch.get("stop_reason") == "stop",
            "has_marker": "ANSWER:" in text,
            "n_chars": len(text),
            "n_answer_markers": text.count("ANSWER:"),
        })

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    summary = {
        "log": path,
        "n": n,
        "accuracy": round(acc, 4),
        "stopped_rate": round(sum(r["stopped"] for r in rows) / n, 4),
        "has_marker_rate": round(sum(r["has_marker"] for r in rows) / n, 4),
        "multi_marker_rate": round(sum(r["n_answer_markers"] > 1 for r in rows) / n, 4),
        "median_chars": sorted(r["n_chars"] for r in rows)[n // 2],
        "wrong_but_stopped": sum(1 for r in rows if not r["correct"] and r["stopped"]),
        "wrong_and_not_stopped": sum(1 for r in rows if not r["correct"] and not r["stopped"]),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "items": rows}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
