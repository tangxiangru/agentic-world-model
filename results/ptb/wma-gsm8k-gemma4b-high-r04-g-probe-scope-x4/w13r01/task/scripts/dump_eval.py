#!/usr/bin/env python3
"""Dump one inspect-ai eval log to a per-item jsonl.

--json-output-file from evaluate.py writes results.scores[0].metrics only, so the
failing-item subset and the format diagnostic have to be read out of the .json log.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ANSWER_LINE = re.compile(r"ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?", re.IGNORECASE)


def latest_log(logdir: str) -> str:
    logs = sorted(glob.glob(os.path.join(logdir, "*.json")), key=os.path.getmtime)
    if not logs:
        raise SystemExit(f"no .json eval log under {logdir}")
    return logs[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="path to the inspect .json log")
    ap.add_argument("--logdir", default="logs", help="dir to take the newest log from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.log or latest_log(args.logdir)
    with open(path) as f:
        log = json.load(f)

    rows = []
    for s in log.get("samples", []):
        completion = ""
        for m in reversed(s.get("messages", [])):
            if m.get("role") == "assistant":
                c = m.get("content")
                completion = c if isinstance(c, str) else "".join(
                    p.get("text", "") for p in c if isinstance(p, dict)
                )
                break
        score = next(iter((s.get("scores") or {}).values()), {})
        out = s.get("output") or {}
        choices = out.get("choices") or []
        stop_reason = (choices[0].get("stop_reason") if choices else None) or out.get(
            "stop_reason"
        )
        usage = out.get("usage") or {}
        rows.append(
            {
                "id": s.get("id"),
                "gold": (s.get("target") if isinstance(s.get("target"), str)
                         else (s.get("target") or [""])[0]),
                "correct": score.get("value") == "C",
                "extracted": score.get("answer"),
                "has_answer_line": bool(ANSWER_LINE.search(completion)),
                "ends_with_answer_line": bool(
                    ANSWER_LINE.search(completion.strip()[-80:] if completion else "")
                ),
                "n_answer_lines": len(ANSWER_LINE.findall(completion)),
                "stop_reason": stop_reason,
                "output_tokens": usage.get("output_tokens"),
                "garbage_prefix": completion.lstrip()[:4] in ("!!!!", "????"),
                "chars": len(completion),
                "completion": completion,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows) or 1
    acc = sum(r["correct"] for r in rows) / n
    print(f"log            {path}")
    print(f"items          {len(rows)}")
    print(f"accuracy       {acc:.4f}")
    print(f"no ANSWER line {sum(not r['has_answer_line'] for r in rows) / n:.4f}")
    print(f"answer not at end {sum(not r['ends_with_answer_line'] for r in rows) / n:.4f}")
    print(f"garbage prefix {sum(r['garbage_prefix'] for r in rows)}")
    print(f"stop_reason    {json.dumps({k: sum(r['stop_reason'] == k for r in rows) for k in {r['stop_reason'] for r in rows}})}")
    toks = [r["output_tokens"] for r in rows if r["output_tokens"]]
    if toks:
        toks.sort()
        print(f"out tokens p50={toks[len(toks)//2]} p95={toks[int(len(toks)*0.95)]} max={toks[-1]}")
    print(f"wrote          {args.out}")


if __name__ == "__main__":
    main()
