#!/usr/bin/env python3
"""Tag the most recent inspect eval log: accuracy, format/termination failures,
and a per-item record so a later card can diff the watch set."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re

ANS_LINE = re.compile(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?\s*$")


def load(log_glob: str):
    f = sorted(glob.glob(log_glob))[-1]
    return f, json.load(open(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/home/ben/task/logs/*_gsm8k_*.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    f, d = load(args.glob)
    S = d["samples"]
    recs = []
    tag = collections.Counter()
    for s in S:
        c = s["output"]["choices"][0]["message"]["content"]
        if isinstance(c, list):
            c = "".join(x.get("text", "") for x in c)
        correct = s["scores"]["match"]["value"] == "C"
        wf = bool(ANS_LINE.search(c.strip()))
        stop = s["output"]["choices"][0].get("stop_reason")
        tag[(correct, wf, stop)] += 1
        recs.append(
            dict(id=s["id"], correct=correct, wellformed=wf, stop=stop,
                 target=s["target"], answer=s["scores"]["match"].get("answer"),
                 n_chars=len(c), tail=c[-300:])
        )
    n = len(recs)
    acc = sum(r["correct"] for r in recs) / n
    fails = [r for r in recs if not r["correct"]]
    malformed = sum(1 for r in fails if not r["wellformed"])
    print("log:", f)
    print(f"n={n} accuracy={acc:.4f}")
    print(f"failures={len(fails)} malformed={malformed} "
          f"share={malformed/max(len(fails),1):.3f}")
    print("truncated (max_tokens):", sum(1 for r in recs if r["stop"] == "max_tokens"))
    for k, v in sorted(tag.items(), key=lambda x: -x[1]):
        print("  (correct,wellformed,stop)=", k, v)
    json.dump({"log": f, "accuracy": acc, "n": n,
               "malformed_share_of_failures": malformed / max(len(fails), 1),
               "records": recs}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
