#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k eval log: accuracy, format compliance, failures."""
from __future__ import annotations

import argparse
import json
import re
import sys

ANS = re.compile(r"^ANSWER:", re.M)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--dump", default=None, help="write per-sample jsonl here")
    ap.add_argument("--show", type=int, default=0, help="print N wrong completions")
    args = ap.parse_args()

    d = json.load(open(args.log))
    samples = d["samples"]
    rows = []
    for s in samples:
        out = s["output"]
        comp = out["choices"][0]["message"]["content"]
        if isinstance(comp, list):
            comp = "".join(c.get("text", "") for c in comp)
        stop = out["choices"][0].get("stop_reason")
        score = list(s["scores"].values())[0]["value"]
        rows.append(
            {
                "id": s["id"],
                "target": s["target"],
                "correct": score == "C",
                "n_answer_lines": len(ANS.findall(comp)),
                "stop_reason": stop,
                "n_out_tokens": out.get("usage", {}).get("output_tokens"),
                "completion": comp,
                "question": s["input"] if isinstance(s["input"], str) else None,
            }
        )

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    exactly_one = sum(r["n_answer_lines"] == 1 for r in rows) / n
    stopped = sum(r["stop_reason"] == "stop" for r in rows) / n
    mean_tok = sum(r["n_out_tokens"] or 0 for r in rows) / n
    print(f"n={n} accuracy={acc:.3f} exactly_one_ANSWER={exactly_one:.3f} "
          f"stop_reason==stop={stopped:.3f} mean_out_tokens={mean_tok:.0f}")
    from collections import Counter
    print("stop_reasons:", Counter(r["stop_reason"] for r in rows))
    print("n_ANSWER lines:", Counter(r["n_answer_lines"] for r in rows).most_common(6))

    if args.dump:
        with open(args.dump, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", args.dump)

    shown = 0
    for r in rows:
        if shown >= args.show:
            break
        if not r["correct"]:
            print("=" * 70)
            print("TARGET", r["target"], "| stop", r["stop_reason"], "| tok", r["n_out_tokens"])
            print(r["completion"][:1500])
            shown += 1


if __name__ == "__main__":
    main()
