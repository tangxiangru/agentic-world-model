#!/usr/bin/env python3
"""Format/termination diagnostic over an inspect-ai eval log."""
from __future__ import annotations

import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect json log; default = newest in logs/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*gsm8k*.json"))[-1]
    d = json.load(open(path))
    ss = d["samples"]
    n = len(ss)
    n_fmt = n_stop = n_correct = 0
    out_tokens = 0
    wrong_examples = []
    for s in ss:
        ch = s["output"]["choices"][0]
        out = ch["message"]["content"]
        if isinstance(out, list):
            out = "".join(c.get("text", "") for c in out)
        lines = [l for l in out.strip().split("\n") if l.strip()]
        if lines and lines[-1].strip().startswith("ANSWER:"):
            n_fmt += 1
        if ch.get("stop_reason") == "stop":
            n_stop += 1
        sc = list(s.get("scores", {}).values())
        ok = bool(sc) and sc[0]["value"] == "C"
        n_correct += ok
        out_tokens += (s.get("model_usage") or {}).get(
            next(iter(s.get("model_usage", {})), ""), {}
        ).get("output_tokens", 0) if s.get("model_usage") else 0
        if not ok and len(wrong_examples) < 25:
            wrong_examples.append({
                "id": s["id"],
                "target": s["target"],
                "answer": sc[0].get("answer") if sc else None,
                "tail": out[-400:],
                "n_lines": len(lines),
            })

    res = {
        "log": path,
        "n": n,
        "accuracy": n_correct / n,
        "format_compliance": n_fmt / n,
        "stopped_cleanly": n_stop / n,
        "mean_output_tokens": out_tokens / n if out_tokens else None,
    }
    json.dump({**res, "wrong_examples": wrong_examples}, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
