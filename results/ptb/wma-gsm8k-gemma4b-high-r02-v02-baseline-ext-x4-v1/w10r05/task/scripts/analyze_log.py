"""Parse an inspect_ai JSON eval log: accuracy, termination/format failures, truncation.

The card's diagnostic cannot come from --json-output-file (metrics only), so it is
computed here from the per-sample completions in logs/*.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import graded_answer  # noqa: E402

ANSWER_LINE = re.compile(r"ANSWER:\s*([^\n]*)")


def completion_of(sample: dict) -> str:
    out = sample.get("output") or {}
    for ch in out.get("choices") or []:
        msg = ch.get("message") or {}
        c = msg.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def stop_reason_of(sample: dict) -> str:
    out = sample.get("output") or {}
    for ch in out.get("choices") or []:
        return ch.get("stop_reason") or ""
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", default=None)
    args = ap.parse_args()

    with open(args.log) as f:
        log = json.load(f)
    samples = log.get("samples") or []

    n = len(samples)
    stats = {
        "n": n,
        "correct": 0,
        "no_answer_line": 0,
        "answer_not_last_number": 0,
        "termination_format_failure": 0,
        "stop_reason_length": 0,
        "garbage_prefix": 0,
        "mean_output_chars": 0.0,
    }
    failures = []
    total_chars = 0
    for s in samples:
        comp = completion_of(s)
        total_chars += len(comp)
        target = s.get("target")
        if isinstance(target, list):
            target = target[0]
        score = (list((s.get("scores") or {}).values()) or [{}])[0]
        ok = score.get("value") == "C"
        stats["correct"] += int(ok)
        if stop_reason_of(s) in ("max_tokens", "length"):
            stats["stop_reason_length"] += 1
        if comp.lstrip().startswith("!!!!"):
            stats["garbage_prefix"] += 1

        m = ANSWER_LINE.findall(comp)
        fmt_fail = False
        if not m:
            stats["no_answer_line"] += 1
            fmt_fail = True
        else:
            stated = graded_answer(m[-1])
            got = graded_answer(comp)
            if stated is None or got is None or stated != got:
                stats["answer_not_last_number"] += 1
                fmt_fail = True
        stats["termination_format_failure"] += int(fmt_fail)

        if not ok and len(failures) < 400:
            failures.append(
                {
                    "id": s.get("id"),
                    "target": target,
                    "format_failure": fmt_fail,
                    "stop_reason": stop_reason_of(s),
                    "completion_tail": comp[-600:],
                    "question": (s.get("input") if isinstance(s.get("input"), str) else "")[:400],
                }
            )

    stats["mean_output_chars"] = round(total_chars / max(n, 1), 1)
    stats["accuracy"] = round(stats["correct"] / max(n, 1), 4)
    stats["termination_format_share"] = round(
        stats["termination_format_failure"] / max(n, 1), 4
    )
    print(json.dumps(stats, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
    if args.dump_failures:
        with open(args.dump_failures, "w") as f:
            for r in failures:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
