"""Read an inspect json eval log and split failures into the categories that
decide what the next card should buy: truncation, no-marker, non-termination
(text after the first ANSWER: line), and plain wrong-number.
"""
import argparse, collections, glob, json, os, re, sys

NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def norm(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f == int(f) else round(f, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.logdir, "*.json")), key=os.path.getmtime)
    if not files:
        sys.exit(f"no json log in {a.logdir}")
    log = json.load(open(files[-1]))
    samples = log.get("samples") or []
    tags = collections.Counter()
    stop_reasons = collections.Counter()
    out_tokens = []
    fails = []
    per_item = {}
    correct = 0
    for s in samples:
        score = list(s["scores"].values())[0]
        ok = score["value"] == "C"
        correct += ok
        msg = s["messages"][-1]
        comp = msg["content"] if isinstance(msg["content"], str) else \
            "".join(c.get("text", "") for c in msg["content"])
        sr = (s.get("output") or {}).get("stop_reason") or \
             ((s.get("output") or {}).get("choices") or [{}])[0].get("stop_reason")
        stop_reasons[sr] += 1
        usage = (s.get("output") or {}).get("usage") or {}
        if usage.get("output_tokens"):
            out_tokens.append(usage["output_tokens"])
        per_item[s["id"]] = bool(ok)
        if ok:
            continue
        i = comp.find("ANSWER:")
        if sr == "max_tokens":
            tag = "truncated"
        elif i < 0:
            tag = "no_marker"
        else:
            tail = comp[i:].split("\n", 1)
            after = tail[1].strip() if len(tail) > 1 else ""
            first_ans = NUM_RE.findall(tail[0])
            got = norm(first_ans[-1]) if first_ans else None
            gold = norm(s["target"] if isinstance(s["target"], str) else s["target"][0])
            if after:
                tag = "kept_going_after_answer" + ("_but_first_answer_right"
                                                   if got == gold else "")
            else:
                tag = "wrong_number"
        tags[tag] += 1
        fails.append({"id": s["id"], "tag": tag, "gold": s["target"],
                      "completion": comp[:3000]})

    n = len(samples)
    res = {"log": files[-1], "n": n, "correct": correct,
           "accuracy": correct / n if n else None,
           "failure_tags": dict(tags), "stop_reasons": dict(stop_reasons),
           "mean_output_tokens": (sum(out_tokens) / len(out_tokens)) if out_tokens else None,
           "max_output_tokens": max(out_tokens) if out_tokens else None}
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump({**res, "per_item": per_item, "failures": fails},
                  open(a.out, "w"), indent=2)
    for f in fails[: a.dump_failures]:
        print("=" * 70)
        print(f["id"], f["tag"], "gold=", f["gold"])
        print(f["completion"][:1200])


if __name__ == "__main__":
    main()
