#!/usr/bin/env python3
"""Apply exp-07's pre-committed artifact rule mechanically.

Written BEFORE the repeat reads returned, so the rule is executed rather than
argued. Rule (from exp-07 hypothesis.decision_rule_precommitted):

  final_model is the candidate with the higher MEAN accuracy across all its
  n=1319 reads. If the two means differ by less than 0.5 pt, keep exp-04,
  because it is already installed and an unnecessary overwrite is pure risk.
  No other quantity may break the tie.

Prints the decision; --install performs the copy, never a move.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import statistics as st
import subprocess

# every n=1319 read of each candidate's weights, including reads taken before
# this card. eval/final_model_full1319.json is exp-04's weights read from the
# byte-identical installed copy, which is what started this card.
READS = {
    "exp-04": ["eval/exp-04_full1319.json", "eval/final_model_full1319.json",
               "eval/exp-04_rep1.json", "eval/exp-04_rep2.json"],
    "exp-06": ["eval/exp-06_full1319.json",
               "eval/exp-06_rep1.json", "eval/exp-06_rep2.json"],
}
CKPT = {"exp-04": "ckpts/exp-04/final", "exp-06": "ckpts/exp-06/final"}
TIE_BAND = 0.005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()

    means, table = {}, {}
    for cand, paths in READS.items():
        vals = []
        for p in paths:
            if not os.path.exists(p):
                print(f"MISSING {p} (excluded)")
                continue
            vals.append(json.load(open(p))["accuracy"])
        assert vals, f"no reads for {cand}"
        means[cand] = st.mean(vals)
        table[cand] = {
            "reads": [round(v, 4) for v in vals], "n_reads": len(vals),
            "mean": round(st.mean(vals), 4),
            "sd": round(st.stdev(vals), 4) if len(vals) > 1 else None,
            "min": round(min(vals), 4), "max": round(max(vals), 4),
        }
        print(f"{cand}: {table[cand]}")

    best = max(means, key=means.get)
    other = [c for c in means if c != best][0]
    gap = means[best] - means[other]
    if gap < TIE_BAND:
        winner, why = "exp-04", (
            f"means differ by {gap*100:.2f} pt, inside the 0.5 pt tie band; "
            f"the rule keeps the already-installed exp-04")
    else:
        winner, why = best, (
            f"{best} mean exceeds {other} by {gap*100:.2f} pt, outside the "
            f"0.5 pt tie band")

    out = {"per_candidate": table, "gap_pt": round(gap * 100, 3),
           "tie_band_pt": TIE_BAND * 100, "winner": winner, "reason": why}
    print(json.dumps(out, indent=2))
    json.dump(out, open("analysis/exp-07_selection.json", "w"), indent=2)

    if args.install:
        src = CKPT[winner]
        cur = subprocess.run(["bash", "-c",
                              "cd final_model && md5sum * | sort -k2 | md5sum"],
                             capture_output=True, text=True).stdout.strip()
        tgt = subprocess.run(["bash", "-c",
                              f"cd {src} && md5sum * | sort -k2 | md5sum"],
                             capture_output=True, text=True).stdout.strip()
        print(f"installed digest {cur}\nwinner    digest {tgt}")
        if cur == tgt:
            print(f"final_model already IS {winner}; no overwrite performed")
        else:
            tmp = "final_model.new"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(src, tmp)
            shutil.rmtree("final_model")
            os.rename(tmp, "final_model")
            print(f"installed {winner} into final_model/ (copy, not move)")


if __name__ == "__main__":
    main()
