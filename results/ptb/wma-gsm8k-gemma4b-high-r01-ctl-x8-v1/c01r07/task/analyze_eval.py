#!/usr/bin/env python3
"""Read an inspect-ai eval log and report accuracy plus the failure modes that
matter for this harness: no ANSWER line, hit the token cap, wrong number.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def newest_log(logdir: str) -> str:
    files = sorted(glob.glob(os.path.join(logdir, "*_gsm8k_*.json")), key=os.path.getmtime)
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", type=int, default=0)
    args = ap.parse_args()

    path = args.log or newest_log(args.logdir)
    d = json.load(open(path))
    samples = d.get("samples") or []
    res = d.get("results") or {}
    metrics = {}
    for s in res.get("scores", []):
        for k, v in s.get("metrics", {}).items():
            metrics[k] = v.get("value")

    n = len(samples)
    correct = no_answer = capped = 0
    fails = []
    for s in samples:
        sc = list(s.get("scores", {}).values())
        ok = bool(sc) and sc[0].get("value") == "C"
        out = ""
        try:
            out = s["output"]["choices"][0]["message"]["content"]
            if isinstance(out, list):
                out = "".join(p.get("text", "") for p in out)
        except Exception:
            pass
        stop = ""
        try:
            stop = s["output"]["choices"][0].get("stop_reason") or ""
        except Exception:
            pass
        if ok:
            correct += 1
        else:
            if "ANSWER:" not in out:
                no_answer += 1
            if stop in ("max_tokens", "length"):
                capped += 1
            if len(fails) < args.dump_failures:
                fails.append({
                    "id": s.get("id"), "target": s.get("target"),
                    "stop_reason": stop, "tail": out[-500:], "chars": len(out),
                })

    summary = {
        "log": path,
        "n": n,
        "accuracy": metrics.get("accuracy"),
        "stderr": metrics.get("stderr"),
        "correct": correct,
        "no_answer_marker_among_failures": no_answer,
        "hit_token_cap_among_failures": capped,
        "mean_completion_chars": (
            sum(len(json.dumps(s.get("output", {}))) for s in samples) / max(1, n)
        ),
    }
    print(json.dumps(summary, indent=2))
    for f in fails:
        print("----", f["id"], "target", f["target"], "stop", f["stop_reason"])
        print(f["tail"])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "failures": fails}, fh, indent=2)


if __name__ == "__main__":
    main()
