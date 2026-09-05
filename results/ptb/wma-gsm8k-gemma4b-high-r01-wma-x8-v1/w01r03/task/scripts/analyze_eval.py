#!/usr/bin/env python3
"""Summarise the newest (or a named) inspect-ai gsm8k log into analysis/<tag>.json.

Reports accuracy, stop-reason mix, answer-marker compliance, completion length,
and the failing items, so a card's diagnostic section points at real evidence.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os


def text_of(sample) -> str:
    c = sample["output"]["choices"][0]["message"]["content"]
    if isinstance(c, list):
        c = "".join(i.get("text", "") for i in c)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect json log; default = newest in logs/")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/home/ben/task/analysis")
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("/home/ben/task/logs/*gsm8k*.json"), key=os.path.getmtime)[-1]
    d = json.load(open(path))
    s = d["samples"]

    sr = collections.Counter()
    n_marker, n_marker_once, lens = 0, 0, []
    fails, wins = [], 0
    for x in s:
        c = text_of(x)
        sr[x["output"]["choices"][0].get("stop_reason")] += 1
        n_marker += "ANSWER:" in c
        n_marker_once += c.count("ANSWER:") == 1
        lens.append(len(c))
        correct = x["scores"]["match"]["value"] == "C"
        wins += correct
        if not correct and len(fails) < 25:
            fails.append(
                {
                    "id": x["id"],
                    "target": x["target"],
                    "stop_reason": x["output"]["choices"][0].get("stop_reason"),
                    "answer_read": x["scores"]["match"].get("answer"),
                    "completion_tail": c[-400:],
                }
            )
    lens.sort()
    out = {
        "log": path,
        "tag": args.tag,
        "n": len(s),
        "accuracy": d["results"]["scores"][0]["metrics"]["accuracy"]["value"],
        "stderr": d["results"]["scores"][0]["metrics"]["stderr"]["value"],
        "stop_reasons": dict(sr),
        "share_max_tokens": round(sr.get("max_tokens", 0) / len(s), 4),
        "share_answer_marker": round(n_marker / len(s), 4),
        "share_answer_marker_exactly_once": round(n_marker_once / len(s), 4),
        "completion_chars_p50": lens[len(lens) // 2],
        "completion_chars_p95": lens[int(len(lens) * 0.95)],
        "failures": fails,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, f"{args.tag}.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
