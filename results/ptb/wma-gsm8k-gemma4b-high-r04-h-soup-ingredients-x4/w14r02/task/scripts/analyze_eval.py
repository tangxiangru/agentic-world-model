#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k log: score, stop reasons, format-failure share."""
import json, re, sys, collections, statistics

def content(x):
    c = x["output"]["choices"][0]["message"]["content"]
    return "".join(p.get("text", "") for p in c) if isinstance(c, list) else c

def main(path, out=None):
    d = json.load(open(path))
    s = d["samples"]
    stop = collections.Counter()
    garbage = noans = notail = 0
    ends_at_first = first_correct = 0
    lens = []
    fails = []
    for x in s:
        stop[x["output"]["choices"][0].get("stop_reason")] += 1
        t = content(x)
        lens.append(len(t))
        if re.match(r'^\s*([!?.,\-*#=_])\1{5,}', t):
            garbage += 1
        if "ANSWER:" not in t:
            noans += 1
        # is the last numeric word on an ANSWER: line?
        m = list(re.finditer(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", t))
        if not m or t[m[-1].end():].strip(" \n*$.") != "":
            notail += 1
        if m:
            # does the completion stop at its FIRST answer, and is that answer right?
            if t[m[0].end():].strip(" \n*$.") == "":
                ends_at_first += 1
            try:
                if abs(float(m[0].group(1).replace(",", "")) - float(x["target"].replace(",", ""))) < 1e-6:
                    first_correct += 1
            except ValueError:
                pass
        v = x["scores"]["match"]["value"]
        if v != "C":
            fails.append({"id": x["id"], "target": x["target"], "tail": t[-300:]})
    r = {
        "path": path,
        "n": len(s),
        "accuracy": d["results"]["scores"][0]["metrics"]["accuracy"]["value"],
        "stderr": d["results"]["scores"][0]["metrics"]["stderr"]["value"],
        "stop_reasons": dict(stop),
        "garbage_prefix": garbage,
        "no_answer_marker": noans,
        "answer_not_last": notail,
        "format_failure_share": round(notail / len(s), 4),
        "ends_at_first_answer_share": round(ends_at_first / len(s), 4),
        "first_answer_accuracy": round(first_correct / len(s), 4),
        "median_chars": statistics.median(lens),
    }
    print(json.dumps(r, indent=2))
    if out:
        json.dump({"summary": r, "failures": fails}, open(out, "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
