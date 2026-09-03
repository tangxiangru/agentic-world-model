#!/usr/bin/env python3
"""Diagnostics over an inspect-ai gsm8k log: termination, answer marker, length."""
from __future__ import annotations

import argparse
import glob
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect json log; default = newest in logs/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = a.log or sorted(glob.glob("logs/*_gsm8k_*.json"))[-1]
    d = json.load(open(p))
    s = d["samples"]
    acc = d["results"]["scores"][0]["metrics"]["accuracy"]["value"]
    stderr = d["results"]["scores"][0]["metrics"]["stderr"]["value"]

    def content(x):
        return x["output"]["choices"][0]["message"]["content"]

    def stopped(x):
        return x["output"]["choices"][0].get("stop_reason") == "stop"

    ok = sum(1 for x in s if stopped(x) and content(x).count("ANSWER:") == 1)
    n_stop = sum(1 for x in s if stopped(x))
    lens = sorted(len(content(x)) for x in s)
    wrong = [
        {"id": x["id"], "target": x["target"], "answer": x["scores"]["match"]["answer"],
         "completion_tail": content(x)[-400:]}
        for x in s if x["scores"]["match"]["value"] != "C"
    ]
    out = {
        "log": p, "n": len(s), "accuracy": acc, "stderr": stderr,
        "stopped": n_stop, "stopped_and_one_marker": ok,
        "chars_p50": lens[len(lens) // 2], "chars_max": lens[-1],
        "n_wrong": len(wrong), "wrong": wrong,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "wrong"}, indent=2))


if __name__ == "__main__":
    main()
