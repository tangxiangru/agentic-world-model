#!/usr/bin/env python3
"""Diagnostics over an inspect-ai gsm8k log: score, termination, answer format."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter


def text_of(sample: dict) -> str:
    t = sample["output"]["choices"][0]["message"]["content"]
    return "".join(c.get("text", "") for c in t) if isinstance(t, list) else t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="inspect log json; default = newest in logs/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.log or max(glob.glob("logs/*gsm8k*.json"), key=os.path.getmtime)
    d = json.load(open(path))
    s = d["samples"]
    stop = Counter()
    correct = first_ok = no_marker = multi_marker = 0
    lens = []
    wrong = []
    for x in s:
        t = text_of(x)
        stop[x["output"]["choices"][0].get("stop_reason")] += 1
        lens.append(len(t))
        n_mark = len(re.findall(r"^ANSWER:", t, re.M))
        if n_mark == 0:
            no_marker += 1
        elif n_mark > 1:
            multi_marker += 1
        ok = x["scores"]["match"]["value"] == "C"
        correct += ok
        m = re.search(r"ANSWER:\s*\$?([\-0-9,\.]+)", t)
        if m:
            try:
                if abs(float(m.group(1).replace(",", "").rstrip(".")) - float(x["target"])) < 1e-6:
                    first_ok += 1
            except ValueError:
                pass
        if not ok and len(wrong) < 8:
            wrong.append({"id": x["id"], "target": x["target"], "tail": t[-400:]})

    n = len(s)
    res = {
        "log": path,
        "n": n,
        "accuracy": correct / n,
        "first_answer_line_correct": first_ok / n,
        "no_answer_marker": no_marker / n,
        "multiple_answer_markers": multi_marker / n,
        "stop_reasons": dict(stop),
        "max_tokens_share": stop.get("max_tokens", 0) / n,
        "completion_chars_p50": sorted(lens)[n // 2],
        "wrong_examples": wrong,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "wrong_examples"}, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
