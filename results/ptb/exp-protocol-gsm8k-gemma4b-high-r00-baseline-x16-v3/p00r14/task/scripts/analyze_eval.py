"""Summarise an inspect eval log: accuracy, completion length, termination, tags.

The point is the failure breakdown, not the score - the score is already in the
--json-output-file. Tags separate the three things that go wrong differently:
the model never stops (run-on), the model stops but never writes the marker
(format), or the model writes a wrong number (arithmetic/reasoning).
"""
import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter


def newest_log(logs_dir):
    files = glob.glob(os.path.join(logs_dir, "*_gsm8k_*.json"))
    return max(files, key=os.path.getmtime)


def tag(text, correct, max_tokens_hit):
    if correct:
        return "correct"
    if max_tokens_hit:
        return "run_on_truncated"
    if "ANSWER:" not in text:
        return "no_answer_marker"
    tail = text[text.rfind("ANSWER:"):]
    if len(tail.strip().split("\n")) > 2:
        return "text_after_answer"
    return "wrong_number"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    path = args.log or newest_log(args.logs_dir)
    d = json.load(open(path))
    samples = d["samples"]
    lens, tags, rows = [], Counter(), []
    for s in samples:
        text = s["output"]["choices"][0]["message"]["content"]
        correct = s["scores"]["match"]["value"] == "C"
        ntok = (s["output"].get("usage") or {}).get("completion_tokens") or 0
        lens.append(len(text))
        t = tag(text, correct, ntok and ntok >= 3900)
        tags[t] += 1
        rows.append({"id": s["id"], "target": s["target"], "correct": correct,
                     "tag": t, "chars": len(text),
                     "answer": s["scores"]["match"].get("answer")})
    acc = sum(r["correct"] for r in rows) / len(rows)
    summary = {"log": path, "n": len(rows), "accuracy": acc,
               "chars_p50": statistics.median(lens), "chars_max": max(lens),
               "tags": dict(tags)}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "items": rows}, f, indent=2)
    for s in samples[: args.show]:
        if s["scores"]["match"]["value"] != "C":
            print("=" * 20, s["id"], "gold", s["target"])
            print(s["output"]["choices"][0]["message"]["content"][-800:])


if __name__ == "__main__":
    main()
