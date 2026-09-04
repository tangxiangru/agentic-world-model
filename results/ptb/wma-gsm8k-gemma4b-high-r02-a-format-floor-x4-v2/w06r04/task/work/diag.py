"""The exp-01 diagnostic, cut the way the grader cuts, over an inspect log.

Reports the two numbers the cards compare:
  end-anchored accuracy  - what the harness scores (last numeric token)
  in-line accuracy       - grading the model's own FIRST "ANSWER: n" line
plus truncation rate, garbage-prefix count (the concurrency guard), and the
share of failures that a termination fix could still recover.

Usage: python work/diag.py <inspect log json> [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import re

ANS = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.M)


def num(x):
    x = x.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(x)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    S = json.load(open(args.log))["samples"]
    n = len(S)
    corr = inline = trunc = garbage = 0
    buckets = collections.Counter()
    toks = []
    for s in S:
        ch = s["output"]["choices"][0]
        comp = ch["message"]["content"]
        toks.append(s["output"]["usage"]["output_tokens"])
        ok = s["scores"]["match"]["value"] == "C"
        corr += ok
        if ch.get("stop_reason") == "max_tokens":
            trunc += 1
        if comp.lstrip().startswith("!!!"):
            garbage += 1
        m = ANS.findall(comp)
        g = num(s["target"])
        first = num(m[0]) if m else None
        iok = first is not None and g is not None and abs(first - g) < 1e-6
        inline += iok
        if not ok:
            if iok:
                buckets["a_first_answer_correct_but_number_after_it"] += 1
            elif not m and ch.get("stop_reason") == "max_tokens":
                buckets["a_truncated_no_answer_line"] += 1
            elif not m:
                buckets["no_answer_line"] += 1
            else:
                buckets["b_genuinely_wrong"] += 1

    wrong = n - corr
    rec = (buckets["a_first_answer_correct_but_number_after_it"]
           + buckets["a_truncated_no_answer_line"])
    out = {
        "log": args.log, "n": n,
        "end_anchored_accuracy": corr / n,
        "in_line_accuracy": inline / n,
        "gap_pp": 100 * (inline - corr) / n,
        "truncation_rate": trunc / n,
        "garbage_prefix_count": garbage,
        "mean_output_tokens": sum(toks) / n,
        "max_output_tokens": max(toks),
        "failure_buckets": dict(buckets),
        "termination_recoverable_share_of_failures": (rec / wrong) if wrong else 0.0,
    }
    print(json.dumps(out, indent=2))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
