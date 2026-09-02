#!/usr/bin/env python3
"""Turn the newest inspect log into the diagnostic numbers the cards ask for."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re


def newest_log(after: float | None = None) -> str:
    logs = sorted(glob.glob("/home/ben/task/logs/*_gsm8k_*.json"), key=os.path.getmtime)
    if after is not None:
        logs = [p for p in logs if os.path.getmtime(p) > after]
    assert logs, "no inspect log found"
    return logs[-1]


def text(sample) -> str:
    c = sample["output"]["choices"][0]["message"]["content"]
    return "".join(p.get("text", "") for p in c) if isinstance(c, list) else c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-failures", default=None)
    args = ap.parse_args()

    path = args.log or newest_log()
    d = json.load(open(path))
    s = d["samples"]

    natural_stop = correct = has_answer = 0
    lens = []
    failures = []
    for x in s:
        t = text(x)
        lens.append(len(t))
        if x["output"]["choices"][0].get("stop_reason") == "stop":
            natural_stop += 1
        if re.search(r"ANSWER:\s*\S", t):
            has_answer += 1
        ok = x["scores"]["match"]["value"] == "C"
        correct += ok
        if not ok:
            failures.append(
                {
                    "id": x["id"],
                    "question": x["input"] if isinstance(x["input"], str) else str(x["input"])[:400],
                    "gold": x["target"],
                    "model_answer": x["scores"]["match"].get("answer"),
                    "tail": t[-400:],
                }
            )

    out = {
        "log": path,
        "n": len(s),
        "accuracy": correct / len(s),
        "natural_stop_frac": natural_stop / len(s),
        "has_answer_frac": has_answer / len(s),
        "median_completion_chars": sorted(lens)[len(lens) // 2],
        "n_failures": len(failures),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    if args.dump_failures:
        with open(args.dump_failures, "w") as f:
            for x in failures:
                f.write(json.dumps(x) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
