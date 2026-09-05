#!/usr/bin/env python3
"""Tag an inspect-ai eval log: which failures are format failures, which are
wrong numbers, and how long the completions are."""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter

ANSWER_RE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)\s*$")


def main(log_dir: str, out: str) -> None:
    path = sorted(glob.glob(f"{log_dir}/*.json"))[-1]
    d = json.load(open(path))
    rows = []
    for s in d["samples"]:
        comp = s["output"]["choices"][0]["message"]["content"]
        if isinstance(comp, list):
            comp = "".join(c.get("text", "") for c in comp)
        text = comp.strip()
        lines = [l for l in text.split("\n") if l.strip()]
        last = lines[-1].strip() if lines else ""
        m = ANSWER_RE.search(last)
        pred = m.group(1).replace(",", "") if m else None
        ok = s["scores"]["match"]["value"] == "C"
        if ok:
            tag = "correct"
        elif pred is None:
            tag = "format_no_answer_line"
        elif s["output"]["choices"][0].get("stop_reason") == "max_tokens":
            tag = "format_truncated"
        else:
            tag = "wrong_number"
        rows.append({
            "id": s["id"], "target": s["target"], "pred": pred, "tag": tag,
            "stop_reason": s["output"]["choices"][0].get("stop_reason"),
            "chars": len(text),
            "question": s["input"] if isinstance(s["input"], str) else None,
            "completion": text,
        })
    json.dump(rows, open(out, "w"), indent=1)
    c = Counter(r["tag"] for r in rows)
    n = len(rows)
    fails = n - c["correct"]
    fmt = c["format_no_answer_line"] + c["format_truncated"]
    print(f"log={path}")
    print(f"n={n} acc={c['correct']/n:.3f} {dict(c)}")
    print(f"failures={fails} format_failures={fmt} "
          f"share={fmt/max(fails,1):.3f}")
    lens = sorted(r["chars"] for r in rows)
    print(f"completion chars p50={lens[n//2]} p95={lens[int(n*.95)]} max={lens[-1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
