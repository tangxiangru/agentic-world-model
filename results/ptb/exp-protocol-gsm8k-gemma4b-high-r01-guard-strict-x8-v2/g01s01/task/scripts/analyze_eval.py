"""Summarise an inspect-ai gsm8k eval log: accuracy, format compliance, length."""
from __future__ import annotations

import argparse
import json
import re
import sys

ANS_LINE = re.compile(r"ANSWER:\s*-?[\d,]+(\.\d+)?\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-wrong", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.log))
    samples = d["samples"]
    n = len(samples)
    correct = 0
    wellformed = 0
    truncated = 0
    lens = []
    wrong = []
    for s in samples:
        sc = list(s["scores"].values())[0]
        ok = sc["value"] == "C"
        correct += ok
        out = s["output"]["choices"][0]["message"]["content"]
        if isinstance(out, list):
            out = "".join(c.get("text", "") for c in out)
        stop = s["output"]["choices"][0].get("stop_reason")
        if stop == "max_tokens":
            truncated += 1
        if ANS_LINE.search(out.strip()):
            wellformed += 1
        lens.append(len(out))
        if not ok:
            wrong.append({"id": s["id"], "target": s["target"], "answer": sc.get("answer"),
                          "stop_reason": stop, "output": out})
    lens.sort()
    res = {
        "n": n,
        "accuracy": correct / n,
        "wellformed_answer_line": wellformed / n,
        "hit_max_tokens": truncated / n,
        "out_chars_p50": lens[n // 2],
        "out_chars_p95": lens[int(n * 0.95)],
        "out_chars_max": lens[-1],
    }
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump({"summary": res, "wrong": wrong}, open(args.out, "w"), indent=2)
        print("wrote", args.out, file=sys.stderr)
    for w in wrong[: args.dump_wrong]:
        print("=" * 70)
        print("id", w["id"], "gold", w["target"], "got", w["answer"], "stop", w["stop_reason"])
        print(w["output"][:1500])


if __name__ == "__main__":
    main()
