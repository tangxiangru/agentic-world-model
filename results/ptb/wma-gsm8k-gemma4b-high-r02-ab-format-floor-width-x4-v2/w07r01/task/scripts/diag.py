"""Diagnostic over an inspect-ai json eval log.

evaluate.py --json-output-file writes only the scorer metrics; the completions live
in the inspect log. This computes what the cards' evaluation.diagnostic promises:
  * format_ok   - share of completions whose LAST non-empty line is 'ANSWER: <number>'
  * cap_hit     - share that stopped for length (never emitted <end_of_turn>)
  * accuracy    - recomputed from the scores, as a cross-check
and dumps the failures so the next card's problem section has real evidence.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ANS_RE = re.compile(r"^ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\.?$")


def newest_log(log_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(log_dir, "*.json")), key=os.path.getmtime)
    if not cands:
        raise SystemExit(f"no inspect logs in {log_dir}")
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-failures", default=None)
    args = ap.parse_args()

    path = args.log or newest_log(args.log_dir)
    with open(path) as f:
        log = json.load(f)

    samples = log.get("samples") or []
    n = len(samples)
    fmt_ok = cap = correct = cap_with_marker = 0
    fails, lens = [], []
    for s in samples:
        out = s.get("output", {})
        choices = out.get("choices") or [{}]
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        stop = choices[0].get("stop_reason")
        usage = out.get("usage") or {}
        lens.append(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        lines = [x for x in content.strip().split("\n") if x.strip()]
        ok = bool(lines) and bool(ANS_RE.match(lines[-1].strip()))
        fmt_ok += ok
        is_cap = stop in ("max_tokens", "model_length")
        cap += is_cap
        # separates "cannot terminate" from "cannot format": a capped completion that
        # still contains the marker means the model knows the format but not the stop token
        cap_with_marker += is_cap and ("ANSWER:" in content)
        score = list((s.get("scores") or {}).values())
        val = score[0].get("value") if score else None
        is_c = val == "C"
        correct += is_c
        if not is_c:
            fails.append({
                "id": s.get("id"),
                "question": (s.get("input") if isinstance(s.get("input"), str) else str(s.get("input")))[:1200],
                "gold": s.get("target"),
                "extracted": score[0].get("answer") if score else None,
                "stop_reason": stop,
                "format_ok": ok,
                "completion_tail": content[-700:],
            })

    res = {
        "log": path,
        "n": n,
        "accuracy": correct / n if n else None,
        "format_ok": fmt_ok / n if n else None,
        "cap_hit": cap / n if n else None,
        "cap_hit_containing_marker": (cap_with_marker / cap) if cap else None,
        "mean_completion_tokens": sum(lens) / n if n else None,
        "max_completion_tokens": max(lens) if lens else None,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    if args.dump_failures:
        with open(args.dump_failures, "w") as f:
            for r in fails:
                f.write(json.dumps(r) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
