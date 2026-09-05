#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k log: score, termination, format, failures."""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter


def norm(x: str):
    x = x.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(x)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect json log; default: newest in logs/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-failures", default=None)
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*gsm8k*.json"))[-1]
    d = json.load(open(path))
    s = d["samples"]

    correct = sum(1 for x in s if x["scores"]["match"]["value"] == "C")
    stops = Counter(x["output"]["choices"][0].get("stop_reason") for x in s)
    n_marker = sum(
        1
        for x in s
        if x["output"]["choices"][0]["message"]["content"].count("ANSWER:") == 1
    )
    lens = sorted(
        len(x["output"]["choices"][0]["message"]["content"]) for x in s
    )

    failures = []
    for x in s:
        if x["scores"]["match"]["value"] == "C":
            continue
        out = x["output"]["choices"][0]["message"]["content"]
        tgt = x["target"] if isinstance(x["target"], str) else x["target"][0]
        failures.append(
            {
                "id": x["id"],
                "question": x["input"] if isinstance(x["input"], str) else str(x["input"])[:400],
                "gold": tgt,
                "model_answer": x["scores"]["match"].get("answer"),
                "stop_reason": x["output"]["choices"][0].get("stop_reason"),
                "completion_tail": out[-500:],
            }
        )

    summary = {
        "log": path,
        "n": len(s),
        "accuracy": correct / len(s),
        "stop_reasons": dict(stops),
        "share_max_tokens": stops.get("max_tokens", 0) / len(s),
        "share_exactly_one_answer_marker": n_marker / len(s),
        "completion_chars_p50": lens[len(lens) // 2],
        "completion_chars_max": lens[-1],
        "n_failures": len(failures),
    }
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    if args.dump_failures:
        json.dump(failures, open(args.dump_failures, "w"), indent=2)
        print(f"wrote {len(failures)} failures to {args.dump_failures}")


if __name__ == "__main__":
    main()
