"""Summarise an inspect_ai gsm8k log: accuracy plus a failure taxonomy.

Buckets a wrong sample as
  no_stop        : the model ran to the token cap (stop_reason != stop)
  no_marker      : the completion never emits the 'ANSWER: ' line the prompt asks for
  trailing_junk  : it does emit 'ANSWER: n' but a later number is what the scorer reads
  wrong_number   : format fine, arithmetic/reasoning wrong
"""
from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import grader_format as gf  # noqa: E402


def main(path: str, out: str | None = None) -> None:
    d = json.load(open(path))
    samples = d.get("samples") or []
    n = len(samples)
    rows, tally = [], {}
    correct = 0
    for s in samples:
        comp = s["output"]["choices"][0]["message"]["content"]
        if isinstance(comp, list):
            comp = "".join(c.get("text", "") for c in comp)
        stop = s["output"]["choices"][0].get("stop_reason")
        gold = s["target"] if isinstance(s["target"], str) else s["target"][0]
        ok = (s["scores"]["match"]["value"] == "C") if s.get("scores") else gf.score_completion(comp, gold)
        correct += ok
        tag = "correct"
        if not ok:
            if stop != "stop":
                tag = "no_stop"
            elif gf.ANSWER_MARKER not in comp:
                tag = "no_marker"
            else:
                after = comp.rsplit(gf.ANSWER_MARKER, 1)[1]
                claimed = re.match(r"\s*\$?(-?[\d,]+(?:\.\d+)?)", after)
                if claimed and gf.score_completion(gf.ANSWER_MARKER + claimed.group(1), gold):
                    tag = "trailing_junk"
                else:
                    tag = "wrong_number"
        tally[tag] = tally.get(tag, 0) + 1
        rows.append({"id": s["id"], "ok": bool(ok), "tag": tag, "gold": gold,
                     "stop_reason": stop, "n_chars": len(comp), "tail": comp[-300:]})
    print(f"{path}\n n={n} accuracy={correct/max(1,n):.4f}")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:4d}  ({v/max(1,n):.1%})")
    if out:
        with open(out, "w") as f:
            json.dump({"path": path, "n": n, "accuracy": correct / max(1, n),
                       "tally": tally, "rows": rows}, f, indent=1)
        print(" wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
