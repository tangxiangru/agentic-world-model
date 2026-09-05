"""Diagnostics on an inspect-ai gsm8k eval log.

Reports accuracy, how many completions hit the token cap without stopping, how
many carry the 'ANSWER:' line, and — against a previous run's per-id record —
how many watch-set items were fixed and how many regressed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def newest_log(d: str = "logs") -> str:
    files = glob.glob(os.path.join(d, "*_gsm8k_*.json"))
    return max(files, key=os.path.getmtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--baseline", default=None, help="per-id correctness json to diff against")
    args = ap.parse_args()

    path = args.log or newest_log()
    d = json.load(open(path))
    s = d["samples"]
    acc = d["results"]["scores"][0]["metrics"]["accuracy"]["value"]

    per_id, tok, cap_ids, no_ans = {}, [], [], 0
    for x in s:
        ok = x["scores"]["match"]["value"] == "C"
        per_id[x["id"]] = ok
        n = list(x["model_usage"].values())[0]["output_tokens"]
        tok.append(n)
        if n >= args.max_tokens:
            cap_ids.append(x["id"])
        if "ANSWER:" not in x["output"]["choices"][0]["message"]["content"]:
            no_ans += 1

    out = {
        "tag": args.tag,
        "log": path,
        "n": len(s),
        "accuracy": acc,
        "hit_cap": len(cap_ids),
        "no_answer_line": no_ans,
        "p50_tokens": statistics.median(tok),
        "p95_tokens": sorted(tok)[int(len(tok) * 0.95)],
    }

    if args.baseline and os.path.exists(args.baseline):
        base = json.load(open(args.baseline))
        fixed = sum(1 for k, v in per_id.items() if v and not base.get(k, False))
        regress = sum(1 for k, v in per_id.items() if not v and base.get(k, False))
        still = sum(1 for k, v in per_id.items() if not v and not base.get(k, False))
        out.update({"baseline": args.baseline, "fixed": fixed, "regressions": regress, "still_failing": still})

    json.dump(per_id, open(f"analysis/{args.tag}_percorrect.json", "w"))
    json.dump(out, open(f"analysis/{args.tag}_diag.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
