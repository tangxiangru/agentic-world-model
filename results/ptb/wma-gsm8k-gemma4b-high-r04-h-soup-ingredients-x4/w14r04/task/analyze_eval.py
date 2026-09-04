#!/usr/bin/env python3
"""Summarise an inspect_ai eval log: accuracy, termination, and where the graded
number came from.  The grader is match(location='end', numeric=True), i.e. the
LAST number in the completion, so a correct chain that keeps talking still scores 0.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter


def load(path: str) -> dict:
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*.json")), key=os.path.getmtime)
        if not cands:
            raise SystemExit(f"no json log under {path}")
        path = cands[-1]
    print("log:", path)
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--dump-wrong", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load(args.log)
    samples = d.get("samples") or []
    n = len(samples)
    correct = 0
    stop = Counter()
    tags = Counter()
    out_toks = []
    wrong = []
    for s in samples:
        sc = list(s.get("scores", {}).values())
        ok = sc and sc[0].get("value") == "C"
        correct += bool(ok)
        msgs = s.get("messages", [])
        completion = ""
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                c = m.get("content")
                completion = c if isinstance(c, str) else json.dumps(c)
                break
        usage = (s.get("model_usage") or {})
        for v in usage.values():
            out_toks.append(v.get("output_tokens", 0))
        sr = (s.get("output") or {}).get("stop_reason")
        if sr is None:
            ch = ((s.get("output") or {}).get("choices") or [{}])[0]
            sr = ch.get("stop_reason")
        stop[str(sr)] += 1

        tail = completion.rstrip()
        if not ok:
            last_ans_line = None
            for line in reversed(tail.split("\n")):
                if "ANSWER:" in line:
                    last_ans_line = line
                    break
            if last_ans_line is None:
                tags["no_ANSWER_marker"] += 1
            else:
                after = tail.split(last_ans_line)[-1]
                if re.search(r"\d", after):
                    tags["number_after_ANSWER_line"] += 1
                else:
                    tags["wrong_arithmetic_or_reasoning"] += 1
            if len(wrong) < args.dump_wrong:
                wrong.append(
                    {
                        "id": s.get("id"),
                        "target": s.get("target"),
                        "completion_tail": tail[-600:],
                    }
                )
        if tail.startswith("!!!!") or tail[:20].count("!") > 5:
            tags["garbage_prefix"] += 1

    print(f"n={n} correct={correct} accuracy={correct / max(n,1):.4f}")
    print("stop_reason:", dict(stop))
    if out_toks:
        o = sorted(out_toks)
        print(
            "output tokens p50", o[len(o) // 2], "p95", o[int(len(o) * 0.95)],
            "max", o[-1], "mean", round(sum(o) / len(o), 1),
        )
    print("failure tags:", dict(tags))
    for w in wrong:
        print("=" * 60)
        print(w["id"], "gold:", w["target"])
        print(w["completion_tail"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "n": n,
                    "correct": correct,
                    "accuracy": correct / max(n, 1),
                    "stop_reason": dict(stop),
                    "failure_tags": dict(tags),
                    "output_tokens_mean": (sum(out_toks) / len(out_toks)) if out_toks else None,
                    "output_tokens_max": max(out_toks) if out_toks else None,
                },
                f,
                indent=2,
            )
        print("wrote", args.out)


if __name__ == "__main__":
    main()
