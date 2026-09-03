#!/usr/bin/env python3
"""Read the newest inspect log and report the diagnostics exp-01 established:
accuracy, termination shape, and the high-concurrency garbage screen."""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect json log; default = newest in logs/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--watch", default=None, help="jsonl of {id, gold} to re-check")
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*gsm8k*.json"))[-1]
    ss = json.load(open(path))["samples"]
    n = len(ss)
    txt = [s["output"]["choices"][0]["message"]["content"] for s in ss]
    stop = [s["output"]["choices"][0]["stop_reason"] for s in ss]
    correct = {s["id"] for s in ss if s["scores"]["match"]["value"] == "C"}

    d = {
        "source_log": path,
        "n": n,
        "accuracy": len(correct) / n,
        "stop_reason": {k: stop.count(k) for k in set(stop)},
        "hit_max_tokens_frac": stop.count("max_tokens") / n,
        "last_line_is_ANSWER_frac": sum(t.rstrip().split("\n")[-1].startswith("ANSWER:") for t in txt) / n,
        "multi_ANSWER_marker": sum(t.count("ANSWER:") > 1 for t in txt),
        "garbage_prefix": sum(bool(re.match(r"^[\W_]{4,}", t)) for t in txt),
        "output_tokens_p50": statistics.median(s["output"]["usage"]["output_tokens"] for s in ss),
        "input_tokens_p50": statistics.median(s["output"]["usage"]["input_tokens"] for s in ss),
    }
    if args.watch:
        watch = [json.loads(l) for l in open(args.watch)]
        ids = {w["id"] for w in watch}
        present = [s for s in ss if s["id"] in ids]
        d["watch_set"] = {
            "n_in_dev_slice": len(present),
            "fixed": sum(1 for s in present if s["id"] in correct),
            "still_failing": sum(1 for s in present if s["id"] not in correct),
        }
        d["watch_set"]["regressions"] = sum(
            1 for s in ss if s["id"] not in ids and s["id"] not in correct
        )
    json.dump(d, open(args.out, "w"), indent=2)
    print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
