#!/usr/bin/env python3
"""Read an inspect_ai per-sample eval log and report the exp-01 diagnostic.

--json-output-file only writes the aggregate metrics dict (evaluate.py L88-96),
so everything per-sample has to come from the inspect log itself
(log_format='json', INSPECT_LOG_DIR defaults to ./logs relative to cwd).

Reports:
  (a) share of completions containing an "ANSWER:" line at all
  (b) among items scored incorrect, share whose FIRST "ANSWER:" number == gold
      (format failure rather than arithmetic failure)
  (c) share of completions whose stop_reason is max_tokens
  (d) completions whose first 40 chars are one character repeated ("!!!!"),
      with the sample index, as a concurrency-corruption check
  (e) the generate config the eval actually served
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ANSWER_LINE = re.compile(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)", re.IGNORECASE)


def norm(x: str) -> str | None:
    x = x.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(x)
    except ValueError:
        return None
    return format(f, ".5g")


def latest_log(logdir: str) -> str:
    cands = sorted(glob.glob(os.path.join(logdir, "*.json")), key=os.path.getmtime)
    cands = [c for c in cands if "gsm8k" in os.path.basename(c)]
    if not cands:
        raise SystemExit(f"no gsm8k eval log in {logdir}")
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-wrong", type=int, default=0)
    args = ap.parse_args()

    path = args.log or latest_log(args.logdir)
    with open(path) as f:
        log = json.load(f)

    report: dict = {"log": path}
    report["accuracy"] = None
    try:
        sc = log["results"]["scores"][0]["metrics"]
        report["accuracy"] = sc["accuracy"]["value"]
        report["stderr"] = sc.get("stderr", {}).get("value")
    except Exception as e:  # pragma: no cover
        report["accuracy_error"] = repr(e)

    # (e) served generate config
    report["eval_config"] = {
        k: v
        for k, v in (log.get("eval", {}).get("model_generate_config") or {}).items()
        if v is not None
    }
    report["model_args"] = log.get("eval", {}).get("model_args")

    samples = log.get("samples") or []
    report["n_samples"] = len(samples)

    n_answer_line = 0
    n_max_tokens = 0
    garbage = []
    wrong = []
    wrong_fmt = 0
    wrong_arith = 0
    lens = []
    correct = 0

    for i, s in enumerate(samples):
        # completion text
        comp = ""
        out = s.get("output") or {}
        for ch in out.get("choices") or []:
            msg = ch.get("message") or {}
            c = msg.get("content")
            if isinstance(c, str):
                comp = c
            elif isinstance(c, list):
                comp = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            if (ch.get("stop_reason") or "") == "max_tokens":
                n_max_tokens += 1
        lens.append(len(comp))
        gold = norm(str(s.get("target")))
        scores = s.get("scores") or {}
        val = None
        for v in scores.values():
            val = v.get("value")
        is_correct = val in ("C", 1, 1.0, True)
        correct += int(is_correct)

        m = ANSWER_LINE.search(comp)
        if m:
            n_answer_line += 1
        body = comp.strip()[:40]
        if len(set(body)) <= 2 and len(body) >= 20:
            garbage.append(i)

        if not is_correct:
            first = norm(m.group(1)) if m else None
            if first is not None and gold is not None and first == gold:
                wrong_fmt += 1
            else:
                wrong_arith += 1
            if len(wrong) < args.dump_wrong:
                wrong.append(
                    {
                        "id": s.get("id"),
                        "gold": gold,
                        "first_answer_line": first,
                        "completion_head": comp[:400],
                        "completion_tail": comp[-400:],
                    }
                )

    n = max(len(samples), 1)
    n_wrong = max(wrong_fmt + wrong_arith, 1)
    report["scored_correct"] = correct
    report["share_with_answer_line"] = n_answer_line / n
    report["share_max_tokens"] = n_max_tokens / n
    report["wrong_but_first_ANSWER_line_correct"] = wrong_fmt
    report["wrong_and_answer_line_also_wrong"] = wrong_arith
    report["format_share_of_failures"] = wrong_fmt / n_wrong
    report["garbage_prefix_indices"] = garbage
    report["completion_chars_mean"] = sum(lens) / n
    report["completion_chars_p95"] = sorted(lens)[int(0.95 * (n - 1))] if lens else None
    report["wrong_examples"] = wrong

    js = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)
    print(js[:6000])


if __name__ == "__main__":
    main()
