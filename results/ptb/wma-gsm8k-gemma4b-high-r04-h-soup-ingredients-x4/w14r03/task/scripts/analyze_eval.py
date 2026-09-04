#!/usr/bin/env python3
"""Diagnostics on an inspect-ai eval log: format compliance and stop behaviour."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ANS_LINE = re.compile(r"^ANSWER:\s*-?[\d,]+(\.\d+)?\s*$", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="path to the inspect .json log")
    ap.add_argument("--logdir", default="/home/ben/task/logs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    path = a.log
    if path is None:
        cands = sorted(glob.glob(os.path.join(a.logdir, "*.json")), key=os.path.getmtime)
        path = cands[-1]
    d = json.load(open(path))

    samples = d.get("samples") or []
    n = len(samples)
    stats = {
        "log": path,
        "n": n,
        "accuracy": None,
        "has_answer_line_anywhere": 0,
        "answer_line_is_last": 0,
        "stop_reason": {},
        "completion_chars_p50": None,
        "wrong_but_has_answer_line": 0,
    }
    if d.get("results"):
        for s in d["results"]["scores"]:
            stats["accuracy"] = s["metrics"]["accuracy"]["value"]

    lens = []
    for s in samples:
        out = s.get("output", {})
        choices = out.get("choices") or []
        text = ""
        if choices:
            msg = choices[0].get("message", {})
            c = msg.get("content")
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):
                text = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            sr = choices[0].get("stop_reason", "?")
            stats["stop_reason"][sr] = stats["stop_reason"].get(sr, 0) + 1
        lens.append(len(text))
        has = bool(ANS_LINE.search(text))
        stats["has_answer_line_anywhere"] += int(has)
        last_line = ""
        for line in reversed(text.strip().splitlines()):
            if line.strip():
                last_line = line.strip()
                break
        is_last = bool(re.match(r"^ANSWER:\s*-?[\d,]+(\.\d+)?$", last_line))
        stats["answer_line_is_last"] += int(is_last)
        sc = (s.get("scores") or {})
        correct = any(v.get("value") == "C" for v in sc.values())
        if has and not correct:
            stats["wrong_but_has_answer_line"] += 1

    lens.sort()
    if lens:
        stats["completion_chars_p50"] = lens[len(lens) // 2]
    for k in ("has_answer_line_anywhere", "answer_line_is_last"):
        stats[k + "_share"] = stats[k] / max(1, n)

    # effective sampling params actually used
    stats["model_args"] = d.get("eval", {}).get("model_args")
    stats["generate_config"] = d.get("plan", {}).get("config") or d.get("eval", {}).get("config")

    json.dump(stats, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
