#!/usr/bin/env python3
"""Parse an inspect_ai json eval log into the per-card diagnostic.

Reports, over the scored samples:
  accuracy
  no_answer_line   completions with no 'ANSWER:' line
  runon            last numeric token of the completion != number on the ANSWER line
  garbage          completion starts with a repeated-punctuation prefix (corrupt read)
  truncated        stop_reason indicates the token budget ran out
and writes the failing items to <out>.
"""
import argparse
import glob
import json
import re
import sys

PUNCT_PREFIX = re.compile(r"^\s*([!?.,;:*#\-=_~/\\]{4,})")
ANSWER_LINE = re.compile(r"ANSWER:\s*([^\n]*)")


def last_number(text):
    for w in reversed(re.split(r"\s+", text.strip())):
        w2 = w.replace(",", "").replace("$", "").rstrip(".:;")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return w2
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to an inspect .json log, or a directory of them")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()

    path = a.log
    if not path.endswith(".json"):
        cands = sorted(glob.glob(path.rstrip("/") + "/*.json"))
        if not cands:
            sys.exit(f"no json logs under {path}")
        path = cands[-1]
    d = json.load(open(path))
    samples = d.get("samples") or []
    n = len(samples)
    stats = dict(n=n, correct=0, no_answer_line=0, runon=0, garbage=0, truncated=0)
    fails = []
    for s in samples:
        comp = ""
        try:
            comp = s["output"]["choices"][0]["message"]["content"]
            if isinstance(comp, list):
                comp = "".join(c.get("text", "") for c in comp)
        except Exception:
            pass
        stop = (s.get("output") or {}).get("choices", [{}])[0].get("stop_reason")
        score = list((s.get("scores") or {}).values())
        ok = bool(score) and score[0].get("value") in ("C", 1, 1.0, True)
        stats["correct"] += ok
        m = ANSWER_LINE.search(comp)
        if not m:
            stats["no_answer_line"] += 1
        else:
            declared = last_number(m.group(1))
            actual = last_number(comp)
            if declared is not None and actual is not None and declared != actual:
                stats["runon"] += 1
        if PUNCT_PREFIX.match(comp):
            stats["garbage"] += 1
        if stop in ("max_tokens", "length"):
            stats["truncated"] += 1
        if not ok:
            fails.append({"id": s.get("id"), "target": s.get("target"),
                          "completion_tail": comp[-700:], "stop_reason": stop})
    stats["accuracy"] = round(stats["correct"] / max(n, 1), 4)
    stats["log"] = path
    print(json.dumps(stats, indent=2))
    for f in fails[: a.show]:
        print("=" * 60)
        print(f["id"], "gold:", f["target"], "stop:", f["stop_reason"])
        print(f["completion_tail"])
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"stats": stats, "failures": fails}, fh, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
