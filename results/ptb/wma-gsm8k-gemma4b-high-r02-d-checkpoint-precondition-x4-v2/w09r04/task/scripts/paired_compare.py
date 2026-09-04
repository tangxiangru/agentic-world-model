"""Paired per-item comparison of two inspect logs scored on the same items.

At n=1319 the marginal standard error is ~0.0125, but the two candidates are
graded on identical items, so the paired (McNemar) count has a much smaller
standard error and can separate gaps the marginal errors call unresolved.
"""
import argparse
import json
import math
from pathlib import Path


def scores(p):
    d = json.loads(Path(p).read_text())
    out = {}
    for s in d["samples"]:
        out[s["id"]] = s["scores"]["match"]["value"] == "C"
    return out


def fmt_stats(p):
    d = json.loads(Path(p).read_text())
    cap = 0
    for s in d["samples"]:
        try:
            if (s["output"]["choices"][0]["stop_reason"] or "") in ("max_tokens", "length"):
                cap += 1
        except Exception:
            pass
    return cap / len(d["samples"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a, b = scores(args.a), scores(args.b)
    ids = sorted(set(a) & set(b))
    assert len(ids) == len(a) == len(b), (len(ids), len(a), len(b))
    win = sum(1 for i in ids if a[i] and not b[i])
    lose = sum(1 for i in ids if b[i] and not a[i])
    n = len(ids)
    delta = (win - lose) / n
    se = math.sqrt(win + lose) / n            # paired se of the difference
    z = (win - lose) / math.sqrt(win + lose) if (win + lose) else 0.0
    print(f"n paired            : {n}")
    print(f"{args.label_a:<20}: {sum(a.values())/n:.4f}   cap share {fmt_stats(args.a):.4f}")
    print(f"{args.label_b:<20}: {sum(b.values())/n:.4f}   cap share {fmt_stats(args.b):.4f}")
    print(f"A gains / A loses   : {win} / {lose}   (discordant {win + lose})")
    print(f"paired delta        : {delta:+.4f}  paired se {se:.4f}  z {z:+.2f}")


if __name__ == "__main__":
    main()
