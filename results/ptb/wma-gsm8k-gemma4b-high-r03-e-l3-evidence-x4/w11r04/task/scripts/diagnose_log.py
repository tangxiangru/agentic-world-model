#!/usr/bin/env python3
"""Read an inspect json log and report why samples failed.

Buckets a failure as:
  no_answer_line   completion has no parseable 'ANSWER: <number>' anywhere
  hit_cap          stop_reason is a length/max-token stop
  rambled          an ANSWER: line exists but is not the last non-empty line
  wrong_number     a clean trailing ANSWER: line with the wrong value
Also counts garbage prefixes ('!!!!'-style) - the high-concurrency corruption
mode - and the share of completions that emit <end_of_turn> (i.e. stop cleanly).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)")


def norm(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    if s.endswith("."):
        s = s[:-1]
    if s.endswith(".0"):
        s = s[:-2]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.log))
    samples = d["samples"]
    tags = Counter()
    stop_reasons = Counter()
    garbage = 0
    out_tokens = []
    failures = []
    n_correct = 0
    for s in samples:
        out = s["output"]
        choice = out["choices"][0]
        text = choice["message"]["content"]
        if isinstance(text, list):
            text = "".join(c.get("text", "") for c in text)
        sr = choice.get("stop_reason")
        stop_reasons[sr] += 1
        usage = out.get("usage") or {}
        out_tokens.append(usage.get("output_tokens", 0))
        if text.lstrip()[:6].count("!") >= 4:
            garbage += 1
        score = list(s["scores"].values())[0]
        correct = score["value"] == "C"
        n_correct += correct
        if correct:
            continue
        lines = [l for l in text.strip().splitlines() if l.strip()]
        last = lines[-1] if lines else ""
        m_last = ANS_RE.search(last)
        m_any = ANS_RE.search(text)
        if not m_any:
            tag = "no_answer_line"
        elif not m_last:
            tag = "rambled"
        else:
            tag = "wrong_number"
        if sr in ("max_tokens", "length") and tag != "wrong_number":
            tag = "hit_cap"
        tags[tag] += 1
        if len(failures) < args.dump_failures:
            failures.append({"id": s["id"], "tag": tag, "stop_reason": sr,
                             "target": s["target"], "tail": text[-400:]})

    n = len(samples)
    n_fail = n - n_correct
    report = {
        "log": args.log,
        "n": n,
        "accuracy": n_correct / n,
        "n_failures": n_fail,
        "failure_tags": dict(tags),
        "failure_tag_share": {k: v / max(1, n_fail) for k, v in tags.items()},
        "stop_reasons": dict(stop_reasons),
        "share_stop_clean": stop_reasons.get("stop", 0) / n,
        "garbage_prefix_samples": garbage,
        "mean_output_tokens": sum(out_tokens) / n,
        "max_output_tokens": max(out_tokens) if out_tokens else 0,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"report": report, "failures": failures}, f, indent=2)


if __name__ == "__main__":
    main()
