"""Read an inspect-ai eval log and report accuracy plus the format diagnostic.

The graded rule is match(location="end", numeric=True): the LAST number of the
completion is the answer. The diagnostic here is the share of completions whose
final line is exactly an 'ANSWER: <number>' line, i.e. the model stopped where
it was trained to stop.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ANSWER_LINE = re.compile(r"^ANSWER:\s*\$?-?[\d,]+(\.\d+)?\.?$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="inspect .json log file or a directory holding them")
    ap.add_argument("--dump-failures", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.log
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*.json")), key=os.path.getmtime)
        path = cands[-1]
    with open(path) as f:
        log = json.load(f)

    samples = log["samples"]
    n = len(samples)
    correct = sum(1 for s in samples if s["scores"]["match"]["value"] == "C")
    compliant = 0
    lengths = []
    failures = []
    for s in samples:
        out = s["output"]["choices"][0]["message"]["content"]
        if isinstance(out, list):
            out = "".join(c.get("text", "") for c in out)
        lengths.append(len(out))
        last = out.strip().splitlines()[-1].strip() if out.strip() else ""
        ok = bool(ANSWER_LINE.match(last))
        compliant += ok
        if s["scores"]["match"]["value"] != "C":
            # deliberately NOT storing the question text: these are benchmark test
            # items and must not end up in any file that could seed training data
            failures.append({
                "id": s["id"],
                "gold": s["target"],
                "answer": s["scores"]["match"].get("answer"),
                "last_line": last,
                "compliant": ok,
                "n_chars": len(out),
                "stop_reason": s["output"].get("stop_reason"),
            })

    res = {
        "log": path,
        "n": n,
        "accuracy": correct / n,
        "format_compliance": compliant / n,
        "mean_chars": sum(lengths) / n,
        "max_chars": max(lengths),
        "n_failures": len(failures),
        "noncompliant_failures": sum(1 for f in failures if not f["compliant"]),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    if args.dump_failures:
        with open(args.dump_failures, "w") as f:
            for x in failures:
                f.write(json.dumps(x) + "\n")
        print("failures ->", args.dump_failures)


if __name__ == "__main__":
    main()
