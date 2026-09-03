#!/usr/bin/env python3
"""Summarise the newest inspect eval log: accuracy, stop behaviour, answer-format health."""
import argparse, glob, json, os, re

ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect json log (default: newest in logs/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-samples", default=None)
    args = ap.parse_args()

    f = args.log or sorted(glob.glob("logs/*_gsm8k_*.json"), key=os.path.getmtime)[-1]
    d = json.load(open(f))
    s = d["samples"]
    correct = sum(1 for x in s if x["scores"]["match"]["value"] == "C")
    clean = 0
    no_ans = 0
    no_stop = 0
    first_ok = 0
    for x in s:
        c = x["output"]["choices"][0]["message"]["content"]
        sr = x["output"]["choices"][0].get("stop_reason")
        hits = ANS.findall(c)
        if sr != "stop":
            no_stop += 1
        if not hits:
            no_ans += 1
        if sr == "stop" and len(hits) == 1:
            clean += 1
        if hits:
            try:
                if abs(float(hits[0].replace(",", "")) - float(x["target"])) < 1e-6:
                    first_ok += 1
            except ValueError:
                pass
    out = {"log": f, "n": len(s), "accuracy": correct / len(s),
           "clean_format_share": clean / len(s), "no_answer_marker": no_ans,
           "did_not_stop": no_stop, "first_answer_line_accuracy": first_ok / len(s)}
    json.dump(out, open(args.out, "w"), indent=1)
    if args.copy_samples:
        json.dump(s, open(args.copy_samples, "w"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
