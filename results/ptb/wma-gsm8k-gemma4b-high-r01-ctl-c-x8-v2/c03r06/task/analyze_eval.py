#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k log: accuracy, termination, first-vs-last answer."""
import argparse
import collections
import json
import re
import sys


def norm(x):
    return x.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", default=None)
    a = ap.parse_args()

    d = json.load(open(a.log))
    s = d["samples"]
    stop = collections.Counter()
    first_ok = graded_ok = 0
    lens = []
    failures = []
    for x in s:
        ch = x["output"]["choices"][0]
        c = ch["message"]["content"]
        stop[ch.get("stop_reason")] += 1
        lens.append(len(c))
        ok = x["scores"]["match"]["value"] == "C"
        graded_ok += ok
        m = re.search(r"ANSWER:\s*([^\n]+)", c)
        f = False
        if m:
            try:
                f = abs(float(norm(m.group(1))) - float(x["target"])) < 1e-6
            except Exception:
                f = False
        first_ok += f
        if not ok:
            failures.append({"id": x["id"], "question": x["input"] if isinstance(x["input"], str) else str(x["input"])[:400],
                             "gold": x["target"], "graded_answer": x["scores"]["match"]["answer"],
                             "stop_reason": ch.get("stop_reason"), "completion": c[-1500:]})
    n = len(s)
    lens.sort()
    out = {
        "n": n,
        "accuracy": graded_ok / n,
        "first_answer_accuracy": first_ok / n,
        "stop_reasons": dict(stop),
        "max_tokens_share": stop.get("max_tokens", 0) / n,
        "completion_chars_p50": lens[n // 2],
        "completion_chars_max": lens[-1],
        "log": a.log,
    }
    print(json.dumps(out, indent=2))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
    if a.dump_failures:
        with open(a.dump_failures, "w") as fh:
            for f in failures:
                fh.write(json.dumps(f) + "\n")
        print(f"wrote {len(failures)} failures to {a.dump_failures}", file=sys.stderr)


if __name__ == "__main__":
    main()
