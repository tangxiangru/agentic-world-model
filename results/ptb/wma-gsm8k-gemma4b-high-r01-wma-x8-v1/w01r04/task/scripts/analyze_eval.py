#!/usr/bin/env python3
"""Read an inspect-ai gsm8k log and report the things the score alone hides:
format compliance, stop reasons, degeneration, and the failing items."""
import argparse
import collections
import glob
import json
import os
import re

ANSWER_LINE = re.compile(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?$")


def latest_log(logdir: str) -> str:
    files = glob.glob(os.path.join(logdir, "*_gsm8k_*.json"))
    return max(files, key=os.path.getmtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--logdir", default="/home/ben/task/logs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-failures", default=None)
    args = ap.parse_args()

    path = args.log or latest_log(args.logdir)
    d = json.load(open(path))
    samples = d["samples"]

    stop = collections.Counter()
    wellformed = 0
    correct = 0
    fail_rows = []
    lens = []
    for x in samples:
        ch = x["output"]["choices"][0]
        stop[ch.get("stop_reason")] += 1
        txt = ch["message"]["content"].strip()
        lens.append(len(txt))
        if ANSWER_LINE.search(txt.split("\n")[-1].strip()):
            wellformed += 1
        sc = x["scores"]["match"]
        if sc["value"] == "C":
            correct += 1
        else:
            fail_rows.append({
                "id": x["id"], "question": x["input"], "gold": x["target"],
                "graded_answer": sc.get("answer"), "completion": txt,
            })

    n = len(samples)
    out = {
        "log": path,
        "n": n,
        "accuracy": correct / n,
        "wellformed_final_answer_line": wellformed / n,
        "stop_reasons": dict(stop),
        "hit_token_cap_share": stop.get("max_tokens", 0) / n,
        "median_completion_chars": sorted(lens)[n // 2],
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))

    if args.dump_failures:
        with open(args.dump_failures, "w") as f:
            for r in fail_rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(fail_rows)} failures -> {args.dump_failures}")


if __name__ == "__main__":
    main()
