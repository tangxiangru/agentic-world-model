#!/usr/bin/env python3
"""Read an inspect-ai json eval log and produce the diagnostic the cards ask for:
accuracy, stop-reason histogram, no-ANSWER-line share among failures, degeneracy
counters, plus a failures file and a watch set.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

ANSWER_LINE = re.compile(r"(?m)^\s*ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\s*\.?\s*$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    log = json.load(open(args.log))
    samples = log["samples"]
    stop_reasons = collections.Counter()
    n_correct = 0
    failures = []
    watch = []
    n_no_answer_line = 0
    n_degenerate = 0
    out_tokens = []

    for s in samples:
        score = list(s["scores"].values())[0]
        ok = score["value"] == "C"
        n_correct += ok
        out = s["output"]
        completion = out["choices"][0]["message"]["content"]
        if isinstance(completion, list):
            completion = "".join(c.get("text", "") for c in completion)
        sr = out["choices"][0].get("stop_reason")
        stop_reasons[sr] += 1
        usage = out.get("usage") or {}
        out_tokens.append(usage.get("output_tokens", usage.get("completion_tokens", 0)))
        tail = completion.rstrip()[-4000:]
        has_line = bool(ANSWER_LINE.search(tail))
        # a degenerate completion: one character class repeated >200 times
        degen = bool(re.search(r"(.{1,3}?)\1{200,}", completion[:6000]))
        n_degenerate += degen
        if not ok:
            n_no_answer_line += not has_line
            failures.append(
                {
                    "id": s["id"],
                    # rule 7: no benchmark question/gold/answer text is stored
                    "has_answer_line": has_line,
                    "stop_reason": sr,
                    "completion_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                }
            )
            watch.append({"id": s["id"]})

    n = len(samples)
    summary = {
        "log": args.log,
        "n": n,
        "accuracy": n_correct / n,
        "n_failures": n - n_correct,
        "no_answer_line_among_failures": n_no_answer_line,
        "no_answer_line_share_of_failures": (n_no_answer_line / max(1, n - n_correct)),
        "degenerate_completions": n_degenerate,
        "stop_reasons": dict(stop_reasons),
        "completion_tokens_mean": sum(out_tokens) / n,
        "completion_tokens_max": max(out_tokens),
    }
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    with open(args.out_prefix + "_failures.json", "w") as f:
        json.dump({"summary": summary, "failures": failures}, f, indent=1)
    with open(args.out_prefix + "_watch.jsonl", "w") as f:
        for w in watch:
            f.write(json.dumps(w) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
