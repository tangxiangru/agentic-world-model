"""Summarise an inspect_ai gsm8k log: score, stop reasons, format failures.

    python analyze_eval.py logs/2026-...json [--dump-wrong analysis/x.json]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics


def text(choice) -> str:
    c = choice["message"]["content"]
    return c if isinstance(c, str) else "".join(x.get("text", "") for x in c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--dump-wrong", default=None)
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*_gsm8k_*.json"))[-1]
    d = json.load(open(path))
    S = d["samples"]

    stop = collections.Counter()
    toks, wrong = [], []
    no_marker = 0
    for s in S:
        ch = s["output"]["choices"][0]
        stop[ch.get("stop_reason")] += 1
        c = text(ch)
        if "ANSWER:" not in c:
            no_marker += 1
        toks.append((s["output"].get("usage") or {}).get("output_tokens") or 0)
        if s["scores"]["match"]["value"] != "C":
            wrong.append({
                "id": s["id"], "target": s["target"],
                "answer": s["scores"]["match"].get("answer"),
                "stop": ch.get("stop_reason"), "completion": c,
            })

    n = len(S)
    acc = sum(1 for s in S if s["scores"]["match"]["value"] == "C") / n
    print(json.dumps({
        "log": path,
        "n": n,
        "accuracy": round(acc, 4),
        "stop_reasons": dict(stop),
        "stopped_on_eot_share": round(stop.get("stop", 0) / n, 4),
        "no_ANSWER_marker": no_marker,
        "out_tokens_mean": round(statistics.mean(toks), 1),
        "out_tokens_max": max(toks),
    }, indent=2))

    if args.dump_wrong:
        with open(args.dump_wrong, "w") as f:
            json.dump(wrong, f, indent=2)
        print("wrote", args.dump_wrong, len(wrong), "items")


if __name__ == "__main__":
    main()
