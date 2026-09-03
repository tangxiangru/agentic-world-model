#!/usr/bin/env python3
"""Summarise an inspect json log: accuracy, format-failure rate, sample outputs."""
import json, re, sys
from pathlib import Path

path = Path(sys.argv[1])
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 3
log = json.loads(path.read_text())
samples = log["samples"]
ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", re.I)
n = len(samples); ncorr = 0; nofmt = 0; trunc = 0
rows = []
for s in samples:
    out = s["output"]["choices"][0]["message"]["content"]
    if isinstance(out, list):
        out = "".join(c.get("text", "") for c in out)
    stop = s["output"]["choices"][0].get("stop_reason")
    score = list(s["scores"].values())[0]["value"] if s.get("scores") else "I"
    ok = score == "C"
    ncorr += ok
    m = ANS.findall(out)
    if not m:
        nofmt += 1
    if stop in ("max_tokens", "model_length"):
        trunc += 1
    rows.append({"id": s["id"], "ok": ok, "stop": stop, "ntok": len(out) // 4,
                 "has_ans": bool(m), "target": s["target"], "out": out})
print(json.dumps({"n": n, "accuracy": round(ncorr / n, 4),
                  "no_answer_line": round(nofmt / n, 4),
                  "hit_max_tokens": round(trunc / n, 4),
                  "mean_out_chars": sum(len(r["out"]) for r in rows) // n}, indent=1))
print("\n=== sample failures ===")
k = 0
for r in rows:
    if r["ok"]:
        continue
    k += 1
    if k > n_show:
        break
    print(f"--- id={r['id']} stop={r['stop']} target={r['target']} has_ans={r['has_ans']}")
    print(r["out"][:1500])
    print("   ...TAIL...", repr(r["out"][-300:]))
