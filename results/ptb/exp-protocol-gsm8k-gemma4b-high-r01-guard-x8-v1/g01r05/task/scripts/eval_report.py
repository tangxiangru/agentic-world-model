#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k log: accuracy, stop-token compliance, watch-set delta."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def latest_log(logs_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(logs_dir, "*_gsm8k_*.json")), key=os.path.getmtime)
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--logs-dir", default="/home/ben/task/logs")
    ap.add_argument("--watch", default=None, help="jsonl of {id} that the comparator got wrong")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.log or latest_log(args.logs_dir)
    d = json.load(open(path))
    ss = d["samples"]
    n = len(ss)

    correct_ids, stopped, has_marker, first_line_ok, out_toks = set(), 0, 0, 0, []
    for s in ss:
        c = s["output"]["choices"][0]["message"]["content"]
        sr = s["output"]["choices"][0].get("stop_reason")
        if sr != "max_tokens":
            stopped += 1
        if "ANSWER:" in c:
            has_marker += 1
        m = ANS_RE.search(c)
        if m:
            try:
                if abs(float(m.group(1).replace(",", "")) - float(s["target"])) < 1e-6:
                    first_line_ok += 1
            except ValueError:
                pass
        if s["scores"]["match"]["value"] == "C":
            correct_ids.add(s["id"])
        u = s["output"].get("usage") or {}
        if u.get("completion_tokens"):
            out_toks.append(u["completion_tokens"])

    rep = {
        "log": path,
        "n": n,
        "accuracy": len(correct_ids) / n,
        "share_stopped_before_cap": stopped / n,
        "share_with_ANSWER_marker": has_marker / n,
        "first_ANSWER_line_accuracy": first_line_ok / n,
        "mean_completion_tokens": (sum(out_toks) / len(out_toks)) if out_toks else None,
    }
    if args.watch:
        watch = [json.loads(l)["id"] for l in open(args.watch)]
        present = [i for i in watch if any(s["id"] == i for s in ss)]
        fixed = sum(1 for i in present if i in correct_ids)
        rep["watch_set"] = {
            "path": args.watch, "n_in_dev": len(present),
            "fixed": fixed, "still_failing": len(present) - fixed,
        }
        prev_correct = {s["id"] for s in ss} - set(watch)
        rep["watch_set"]["regressions"] = sum(1 for i in prev_correct if i not in correct_ids)

    json.dump(rep, open(args.out, "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
