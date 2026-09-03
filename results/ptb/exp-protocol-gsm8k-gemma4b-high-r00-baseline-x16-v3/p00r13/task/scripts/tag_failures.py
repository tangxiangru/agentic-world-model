#!/usr/bin/env python3
"""Recompute the exp-01 non-termination diagnostic for the newest inspect log."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

tag = sys.argv[1]
logs = sorted(glob.glob("logs/*gsm8k*.json"), key=os.path.getmtime)
p = logs[-1]
d = json.load(open(p))
ss = d["samples"]
fails = [s for s in ss if s["scores"]["match"]["value"] != "C"]
after = 0
for s in fails:
    c = s["scores"]["match"]["explanation"] or ""
    i = c.find("ANSWER:")
    rest = c[i:].split("\n", 1)[1:] if i >= 0 else []
    if rest and rest[0].strip():
        after += 1
out = {
    "tag": tag,
    "n": len(ss),
    "correct": len(ss) - len(fails),
    "accuracy": (len(ss) - len(fails)) / len(ss),
    "failed": len(fails),
    "failed_with_text_after_first_ANSWER_line": after,
    "nontermination_share_of_failures": after / len(fails) if fails else 0.0,
    "eval_log": p,
}
Path("analysis").mkdir(exist_ok=True)
Path(f"analysis/{tag}_failure_tags.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
