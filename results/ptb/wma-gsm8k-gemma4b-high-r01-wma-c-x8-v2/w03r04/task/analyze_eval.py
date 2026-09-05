#!/usr/bin/env python3
"""Parse the most recent (or a named) inspect-ai gsm8k log into a diagnostic JSON.

Reports: accuracy, share of completions that stop cleanly, share whose last
non-empty line is a well-formed 'ANSWER: N', mean completion tokens, and the
ids/questions of the items that were scored incorrect.
"""
from __future__ import annotations

import argparse
import glob
import json
import re


def text_of(choice) -> str:
    c = choice["message"]["content"]
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*_gsm8k_*.json"))[-1]
    d = json.load(open(path))
    samples = d["samples"]
    n = len(samples)
    wf = stop = 0
    toks = []
    wrong = []
    for s in samples:
        ch = s["output"]["choices"][0]
        out = text_of(ch)
        lines = [l for l in out.strip().split("\n") if l.strip()]
        if lines and re.match(r"^ANSWER:\s*[-\d,.$]+\s*$", lines[-1].strip()):
            wf += 1
        if ch.get("stop_reason") == "stop":
            stop += 1
        u = s["output"].get("usage") or {}
        toks.append(u.get("completion_tokens") or u.get("output_tokens"))
        if s["scores"]["match"]["value"] != "C":
            wrong.append(
                {
                    "id": s["id"],
                    "question": s["input"] if isinstance(s["input"], str) else str(s["input"])[:400],
                    "gold": s["target"],
                    "read": s["scores"]["match"].get("answer"),
                    "tail": out[-300:],
                }
            )
    toks = [t for t in toks if t]
    acc = sum(1 for s in samples if s["scores"]["match"]["value"] == "C") / n
    res = {
        "log": path,
        "n": n,
        "accuracy": acc,
        "wellformed_final_line_share": wf / n,
        "clean_stop_share": stop / n,
        "mean_completion_tokens": sum(toks) / max(1, len(toks)),
        "max_completion_tokens": max(toks) if toks else None,
        "n_wrong": len(wrong),
    }
    print(json.dumps(res, indent=1))
    res["wrong"] = wrong
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
