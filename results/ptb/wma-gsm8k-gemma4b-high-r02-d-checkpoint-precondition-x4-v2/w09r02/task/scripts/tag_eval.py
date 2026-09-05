#!/usr/bin/env python3
"""Tag an inspect-ai gsm8k eval log: correct / runaway / malformed / wrong number."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re

import numpy as np

WELL_FORMED = re.compile(r"^ANSWER:\s*[-$]?[\d,]+(\.\d+)?$")


def tag_log(log_dir: str, out: str):
    f = sorted(glob.glob(f"{log_dir.rstrip('/')}/*.json"))[-1]
    d = json.load(open(f))
    tags = collections.Counter()
    recs = []
    toks = []
    hit_cap = 0
    for s in d["samples"]:
        c = s["output"]["choices"][0]["message"]["content"]
        ok = s["scores"]["match"]["value"] == "C"
        nt = s["output"]["usage"]["output_tokens"]
        toks.append(nt)
        hit_cap += nt >= 3990
        lines = [l for l in c.strip().split("\n") if l.strip()]
        last = lines[-1].strip() if lines else ""
        wellformed = bool(WELL_FORMED.match(last))
        runaway = ("Solve the following math problem step by step" in c) or c.count("ANSWER:") > 1
        if ok:
            t = "correct"
        elif runaway:
            t = "runaway_or_multi_answer"
        elif not wellformed:
            t = "malformed_final_line"
        else:
            t = "wrong_number"
        tags[t] += 1
        recs.append(
            {
                "id": s["id"],
                "tag": t,
                "answer": s["scores"]["match"]["answer"],
                "target": s["target"],
                "out_tokens": nt,
            }
        )
    n = len(d["samples"])
    n_wrong = n - tags["correct"]
    fmt_share = (tags["runaway_or_multi_answer"] + tags["malformed_final_line"]) / n_wrong if n_wrong else 0.0
    summary = {
        "log": f,
        "n": n,
        "accuracy": tags["correct"] / n,
        "tags": dict(tags),
        "format_failure_share_of_failures": fmt_share,
        "mean_out_tokens": float(np.mean(toks)),
        "p95_out_tokens": float(np.percentile(toks, 95)),
        "hit_cap": hit_cap,
        "records": recs,
    }
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tag_log(a.log_dir, a.out)
