"""Score + failure-tag one inspect json log the same way for every card.

Writes {metrics, tag counts, per-item tags} so cards compare like with like.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ANSWER_LINE = re.compile(r"^ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\.?$")


def latest_log(after: float | None = None) -> str:
    fs = glob.glob("logs/*gsm8k*.json")
    if after:
        fs = [f for f in fs if os.path.getmtime(f) >= after]
    return sorted(fs, key=os.path.getmtime)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag-out", default=None)
    args = ap.parse_args()

    path = args.log or latest_log()
    d = json.load(open(path))
    assert d["status"] == "success", d["status"]
    m = d["results"]["scores"][0]["metrics"]
    s = d["samples"]
    counts = {"correct": 0, "format_or_stop": 0, "wrong_number": 0}
    items = []
    for x in s:
        comp = x["scores"]["match"]["explanation"] or ""
        ok = x["scores"]["match"]["value"] == "C"
        last = comp.strip().split("\n")[-1].strip()
        fmt_ok = bool(ANSWER_LINE.match(last))
        tag = "correct" if ok else ("format_or_stop" if not fmt_ok else "wrong_number")
        counts[tag] += 1
        items.append(
            {
                "id": x["id"],
                "tag": tag,
                "target": x["target"],
                "answer": x["scores"]["match"].get("answer"),
                "n_chars": len(comp),
            }
        )
    nf = counts["format_or_stop"] + counts["wrong_number"]
    out = {
        "accuracy": m["accuracy"]["value"],
        "stderr": m["stderr"]["value"],
        "n": len(s),
        "log": path,
        "counts": counts,
        "format_share_of_failures": round(counts["format_or_stop"] / nf, 4) if nf else 0.0,
        "mean_completion_chars": round(sum(i["n_chars"] for i in items) / len(items), 1),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    if args.tag_out:
        json.dump({**out, "items": items}, open(args.tag_out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
