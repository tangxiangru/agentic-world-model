#!/usr/bin/env python3
"""Read an inspect-ai json eval log and report the degeneration checks
the WMA verdict asked for, plus a failure tag per wrong item."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def load(logdir: str) -> dict:
    files = sorted(glob.glob(os.path.join(logdir, "*.json")), key=os.path.getmtime)
    files = [f for f in files if not f.endswith(".summary.json")]
    if not files:
        raise SystemExit(f"no json log in {logdir}")
    with open(files[-1]) as fh:
        return json.load(fh), files[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--dump-wrong", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log, path = load(args.logdir)
    samples = log.get("samples") or []
    acc = None
    try:
        acc = log["results"]["scores"][0]["metrics"]["accuracy"]["value"]
    except Exception:
        pass

    n = len(samples)
    no_answer = capped = 0
    toks = []
    wrong = []
    for s in samples:
        out = s["output"]
        comp = out["choices"][0]["message"]["content"]
        if isinstance(comp, list):
            comp = "".join(c.get("text", "") for c in comp)
        stop = out["choices"][0].get("stop_reason")
        u = out.get("usage") or {}
        toks.append(u.get("output_tokens") or u.get("completion_tokens") or 0)
        if "ANSWER:" not in comp:
            no_answer += 1
        if stop in ("max_tokens", "length"):
            capped += 1
        val = s["scores"]["match"]["value"] if "match" in s.get("scores", {}) else None
        if val != "C":
            wrong.append(
                {
                    "id": s["id"],
                    "question": s["input"] if isinstance(s["input"], str) else None,
                    "gold": s["target"],
                    "answer": s["scores"]["match"].get("answer"),
                    "completion_tail": comp[-400:],
                    "stop_reason": stop,
                    "tokens": u.get("output_tokens"),
                }
            )

    cap_idx = [i for i, s in enumerate(samples)
               if (s["output"]["choices"][0].get("stop_reason") in ("max_tokens", "length"))]
    mean_tok = sum(toks) / max(1, len(toks))
    report = {
        "log": path,
        "n": n,
        "accuracy": acc,
        "no_answer_line": no_answer,
        "no_answer_share": round(no_answer / max(1, n), 4),
        "hit_token_cap": capped,
        "cap_share": round(capped / max(1, n), 4),
        "mean_completion_tokens": round(mean_tok, 1),
        "max_completion_tokens": max(toks) if toks else 0,
        "n_wrong": len(wrong),
        "cap_first_idx": cap_idx[:8],
        "cap_last_idx": cap_idx[-8:],
        "cap_second_half_share": (round(sum(1 for i in cap_idx if i >= n / 2) / len(cap_idx), 3)
                                  if cap_idx else None),
    }
    print(json.dumps(report, indent=1))
    if args.dump_wrong:
        for w in wrong[: args.dump_wrong]:
            print("=" * 70)
            print(json.dumps(w, indent=1)[:2000])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"report": report, "wrong": wrong}, fh, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
