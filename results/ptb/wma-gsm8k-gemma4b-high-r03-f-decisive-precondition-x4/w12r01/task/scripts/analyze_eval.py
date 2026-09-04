#!/usr/bin/env python3
"""Read an inspect-ai json eval log and print the diagnostics the cards ask for."""
import collections
import json
import sys

import numpy as np


def completion(sample):
    c = sample["output"]["choices"][0]["message"]["content"]
    if isinstance(c, list):
        c = "".join(x.get("text", "") for x in c)
    return c


def main(path, dump_wrong=0):
    d = json.load(open(path))
    S = d["samples"]
    stops = collections.Counter()
    n_ans = n_bang = n_correct = 0
    lens = []
    wrong = []
    for s in S:
        c = completion(s)
        stops[s["output"]["choices"][0].get("stop_reason")] += 1
        lens.append(len(c))
        if "ANSWER:" in c:
            n_ans += 1
        if c.strip()[:5].count("!") >= 3:
            n_bang += 1
        sc = list(s["scores"].values())[0]
        ok = sc["value"] == "C"
        n_correct += ok
        if not ok:
            wrong.append((s["id"], s["input"] if isinstance(s["input"], str) else "", s["target"], sc.get("answer"), c))
    n = len(S)
    print(f"file={path}")
    print(f"n={n} accuracy={n_correct/n:.4f} correct={n_correct}")
    print(f"stop_reason={dict(stops)}  stop_share={stops.get('stop',0)/n:.3f}")
    print(f"has_ANSWER={n_ans}/{n}  bang_prefix={n_bang}")
    print(f"chars p50={int(np.percentile(lens,50))} p95={int(np.percentile(lens,95))} max={max(lens)}")
    for i, (sid, q, tgt, ans, c) in enumerate(wrong[:dump_wrong]):
        print(f"\n--- WRONG {i} id={sid} gold={tgt} extracted={ans}\nQ: {q[:300]}\nOUT(tail): ...{c[-500:]}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
