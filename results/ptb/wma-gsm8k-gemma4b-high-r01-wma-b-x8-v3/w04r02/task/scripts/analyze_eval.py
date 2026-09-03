#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k eval log: accuracy, format/termination failures."""
from __future__ import annotations

import argparse
import json
import re
import sys

ANSWER_LINE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", default=None)
    ap.add_argument("--n-show", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.log))
    samples = d["samples"]
    rows, n_ok = [], 0
    for s in samples:
        completion = s["output"]["choices"][0]["message"]["content"]
        if isinstance(completion, list):
            completion = "".join(c.get("text", "") for c in completion)
        stop = s["output"]["choices"][0].get("stop_reason")
        correct = s["scores"]["match"]["value"] == "C"
        n_ok += correct
        text = completion.strip()
        has_answer_line = bool(ANSWER_LINE.search(text))
        nums = NUM.findall(text)
        rows.append(
            {
                "id": s["id"],
                "correct": correct,
                "stop_reason": stop,
                "has_answer_line": has_answer_line,
                "n_tokens": s["output"].get("usage", {}).get("output_tokens"),
                "last_number": nums[-1] if nums else None,
                "target": s["target"],
                "question": s["input"] if isinstance(s["input"], str) else None,
                "completion": completion,
            }
        )

    n = len(rows)
    acc = n_ok / n
    no_fmt = sum(1 for r in rows if not r["has_answer_line"])
    trunc = sum(1 for r in rows if r["stop_reason"] not in ("stop", None))
    fails = [r for r in rows if not r["correct"]]
    fail_no_fmt = sum(1 for r in fails if not r["has_answer_line"])
    summary = {
        "n": n,
        "accuracy": acc,
        "no_answer_line": no_fmt,
        "no_answer_line_share": no_fmt / n,
        "non_stop_finish": trunc,
        "non_stop_share": trunc / n,
        "failures": len(fails),
        "failures_with_format_problem": fail_no_fmt,
        "mean_output_tokens": sum(r["n_tokens"] or 0 for r in rows) / n,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump({"summary": summary, "rows": [{k: v for k, v in r.items() if k != "completion"} for r in rows]}, open(args.out, "w"), indent=2)
    if args.dump_failures:
        with open(args.dump_failures, "w") as fh:
            for r in fails:
                fh.write(json.dumps({"id": r["id"], "question": r["question"], "gold": r["target"], "model_output": r["completion"][:4000]}, ensure_ascii=False) + "\n")
    for r in fails[: args.n_show]:
        print("=" * 70, file=sys.stderr)
        print("ID", r["id"], "gold", r["target"], "stop", r["stop_reason"], file=sys.stderr)
        print(r["completion"][:1500], file=sys.stderr)


if __name__ == "__main__":
    main()
