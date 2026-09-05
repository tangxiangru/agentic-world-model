"""Read an inspect_ai json eval log and split the failures.

The counting rule is fixed before any transcript is read (exp-01 verdict,
precondition 4):

  format_stop  the completion has an "ANSWER: <n>" line whose <n> equals the
               gold answer, but the last number-like token of the whole
               completion is a different one - the grader reads that one -
               OR the completion hit the max_tokens cap (stop_reason
               max_tokens / length), so it never got to an answer at all.
  no_answer    no "ANSWER:" line anywhere and the run stopped normally.
  reasoning    everything else: the model stopped cleanly, put a number on the
               ANSWER line, and that number is wrong.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

ANS_LINE = re.compile(r"ANSWER:\s*([^\n]*)")


def norm(s):
    s = s.strip().strip(".,:;!?)\"'*").replace(",", "").replace("$", "").replace("%", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def last_number(text):
    for w in reversed(re.split(r"\s+", text.strip())):
        n = norm(w)
        if n is not None:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", type=int, default=0)
    args = ap.parse_args()

    log = json.load(open(args.log))
    samples = log.get("samples") or []
    tally = {"format_stop": 0, "no_answer": 0, "reasoning": 0}
    n_correct = 0
    out_tokens = []
    stop_reasons = {}
    failures = []
    for s in samples:
        gold = norm(str(s["target"]))
        msgs = s["messages"]
        completion = ""
        for m in reversed(msgs):
            if m["role"] == "assistant":
                c = m["content"]
                completion = c if isinstance(c, str) else "".join(
                    x.get("text", "") for x in c if isinstance(x, dict))
                break
        usage = (s.get("model_usage") or {})
        for v in usage.values():
            out_tokens.append(v.get("output_tokens", 0))
        sr = (s.get("output") or {}).get("stop_reason") or (
            ((s.get("output") or {}).get("choices") or [{}])[0].get("stop_reason"))
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
        score = list(s["scores"].values())[0]
        ok = score["value"] == "C"
        if ok:
            n_correct += 1
            continue
        hits = [norm(x) for x in ANS_LINE.findall(completion)]
        hits = [h for h in hits if h is not None]
        ln = last_number(completion)
        if sr in ("max_tokens", "length") or (gold in hits and ln != gold):
            tally["format_stop"] += 1
            tag = "format_stop"
        elif not hits:
            tally["no_answer"] += 1
            tag = "no_answer"
        else:
            tally["reasoning"] += 1
            tag = "reasoning"
        failures.append({
            "id": s["id"], "gold": gold, "tag": tag, "stop_reason": sr,
            "answer_lines": hits[:5], "last_number": ln,
            "question": (msgs[0]["content"] if isinstance(msgs[0]["content"], str)
                         else "")[-400:],
            "completion_tail": completion[-500:],
        })

    n = len(samples)
    res = {
        "log": args.log,
        "n": n,
        "accuracy": n_correct / n if n else None,
        "failures": n - n_correct,
        "tally": tally,
        "format_stop_share_of_failures": (
            tally["format_stop"] / (n - n_correct) if n - n_correct else None),
        "mean_output_tokens": sum(out_tokens) / len(out_tokens) if out_tokens else None,
        "max_output_tokens": max(out_tokens) if out_tokens else None,
        "stop_reasons": stop_reasons,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": res, "failures": failures}, f, indent=2)
    for f_ in failures[: args.dump_failures]:
        print("=" * 70, file=sys.stderr)
        print(json.dumps(f_, indent=2)[:2500], file=sys.stderr)


if __name__ == "__main__":
    main()
