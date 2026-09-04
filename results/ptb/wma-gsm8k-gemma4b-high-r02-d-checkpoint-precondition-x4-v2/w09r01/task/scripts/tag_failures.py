#!/usr/bin/env python3
"""Tag an inspect eval log by failure mode, the same way exp-01 did.

Categories, in priority order:
  no_termination_hit_cap            stop_reason == max_tokens
  invented_further_problems         terminated but >1 'ANSWER:' in the completion
  malformed_answer_line             terminated, one marker, final line not 'ANSWER: <number>'
  wellformed_but_wrong              terminated, well formed, wrong number
  correct                           scored C

'format_failure_share' is the first three over n -- the number exp-01 put at
0.9067 for the parent and the diagnostic every later card is compared on.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

ANSWER_LINE = re.compile(r"ANSWER:\s*-?[\d,]+(\.\d+)?")


def completion_text(sample):
    ch = sample["output"]["choices"][0]
    txt = ch["message"]["content"]
    if isinstance(txt, list):
        txt = "".join(c.get("text", "") for c in txt)
    return txt, ch.get("stop_reason")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="inspect json log, or a dir holding one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--watch-out", default=None)
    args = ap.parse_args()

    path = args.log
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*.json")))
        path = cands[-1]
    d = json.load(open(path))
    ss = d["samples"]

    tags = collections.Counter()
    items = []
    for s in ss:
        txt, sr = completion_text(s)
        correct = s["scores"]["match"]["value"] == "C"
        n_marker = txt.count("ANSWER:")
        last = txt.strip().split("\n")[-1].strip()
        if correct:
            tag = "correct"
        elif sr == "max_tokens":
            tag = "no_termination_hit_cap"
        elif n_marker > 1:
            tag = "invented_further_problems"
        elif not ANSWER_LINE.fullmatch(last):
            tag = "malformed_answer_line"
        else:
            tag = "wellformed_but_wrong"
        tags[tag] += 1
        items.append({"id": s["id"], "tag": tag, "correct": correct, "stop_reason": sr,
                      "n_answer_markers": n_marker})

    n = len(ss)
    fmt = sum(v for k, v in tags.items()
              if k in ("no_termination_hit_cap", "invented_further_problems", "malformed_answer_line"))
    out = {
        "eval_log": path,
        "n": n,
        "accuracy": round(sum(i["correct"] for i in items) / n, 4),
        "failure_taxonomy": dict(tags),
        "format_failure_share": round(fmt / n, 4),
        "items": items,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    if args.watch_out:
        with open(args.watch_out, "w") as f:
            for i in items:
                if not i["correct"]:
                    f.write(json.dumps({"id": i["id"]}) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
