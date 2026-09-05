"""Read the newest inspect log and report the failure-mode diagnostics the cards use."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect json log; default = newest in logs/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.log
    if path is None:
        cands = glob.glob("/home/ben/task/logs/*_gsm8k_*.json")
        path = max(cands, key=os.path.getmtime)

    d = json.load(open(path))
    s = d["samples"]
    corr = sum(1 for x in s if x["scores"]["match"]["value"] == "C")
    multi = sum(1 for x in s if x["scores"]["match"]["explanation"].lower().count("answer:") > 1)
    none_marker = sum(1 for x in s if "answer:" not in x["scores"]["match"]["explanation"].lower())
    toks = sorted(x["output"]["usage"]["output_tokens"] for x in s)
    at_cap = sum(1 for t in toks if t >= 3990)

    wrong = [x for x in s if x["scores"]["match"]["value"] != "C"]
    out = {
        "log": path,
        "n": len(s),
        "accuracy": corr / len(s),
        "completions_with_multiple_answer_lines": multi,
        "completions_with_no_answer_line": none_marker,
        "completions_at_token_cap": at_cap,
        "output_tokens_p50": toks[len(toks) // 2],
        "output_tokens_p90": toks[int(0.9 * len(toks))],
        "output_tokens_max": toks[-1],
        "wrong_ids": [x["id"] for x in wrong],
        "wrong_examples": [
            {"id": x["id"], "target": x["target"],
             "model_answer": x["scores"]["match"].get("answer"),
             "tail": x["scores"]["match"]["explanation"][-400:]}
            for x in wrong[:8]
        ],
    }
    json.dump(out, open(args.out, "w"), indent=2)
    for k in ("log", "n", "accuracy", "completions_with_multiple_answer_lines",
              "completions_with_no_answer_line", "completions_at_token_cap",
              "output_tokens_p50", "output_tokens_max"):
        print(f"{k}: {out[k]}")


if __name__ == "__main__":
    main()
