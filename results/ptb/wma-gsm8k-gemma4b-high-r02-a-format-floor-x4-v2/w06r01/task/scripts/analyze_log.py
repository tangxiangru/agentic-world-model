#!/usr/bin/env python3
"""Per-sample diagnostics from an inspect_ai json eval log.

Reports: accuracy, format/termination failure rate (final line is not
"ANSWER: <number>"), stop reasons, output-token distribution against the
--max-tokens cap, and the '!!!!' garbage-prefix count the WMA flagged.
"""
import json
import re
import sys
from collections import Counter

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
log = json.load(open(path))

samples = log.get("samples") or []
ANSWER_LAST = re.compile(r"ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\s*\.?\s*$")

n = len(samples)
correct = bad_fmt = garbage = 0
stop = Counter()
toks = []
fails = []
for s in samples:
    sc = list(s.get("scores", {}).values())
    ok = bool(sc) and sc[0].get("value") == "C"
    correct += ok
    msg = s["messages"][-1]
    txt = msg.get("content")
    if isinstance(txt, list):
        txt = "".join(c.get("text", "") for c in txt if isinstance(c, dict))
    txt = txt or ""
    if not ANSWER_LAST.search(txt.strip()):
        bad_fmt += 1
    if txt.lstrip().startswith("!!!!"):
        garbage += 1
    o = s.get("output") or {}
    stop[(o.get("choices") or [{}])[0].get("stop_reason", "?")] += 1
    u = o.get("usage") or {}
    toks.append(u.get("completion_tokens", 0))
    if not ok and len(fails) < 8:
        fails.append({"id": str(s.get("id")), "target": s.get("target"),
                      "tail": txt.strip()[-400:], "n_tok": u.get("completion_tokens")})

toks.sort()
res = {
    "n": n,
    "accuracy": round(correct / max(1, n), 4),
    "format_fail_rate": round(bad_fmt / max(1, n), 4),
    "garbage_prefix": garbage,
    "stop_reasons": dict(stop),
    "completion_tokens": {"p50": toks[n // 2] if n else 0,
                          "p95": toks[int(.95 * n)] if n else 0,
                          "max": toks[-1] if n else 0},
}
print(json.dumps(res, indent=2))
if out:
    json.dump({**res, "failure_examples": fails}, open(out, "w"), indent=2)
    print("wrote", out)
