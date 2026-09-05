#!/usr/bin/env python3
"""Promote a scored checkpoint into final_model/ behind a regression guard.

Never overwrites final_model/ with something that scored worse than what is
already there: the incumbent's score lives in final_model/PROMOTED.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

FINAL = "/home/ben/task/final_model"
STAMP = os.path.join(FINAL, "PROMOTED.json")

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="a packaged, loadable model dir")
ap.add_argument("--score", type=float, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--card", required=True)
ap.add_argument("--eval-path", required=True)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

if os.path.exists(STAMP):
    cur = json.load(open(STAMP))
    if cur["n"] != a.n and not a.force:
        # protocol rule 2: a number from a different --limit is not a comparator
        print(f"REFUSED: incumbent {cur['card']} was scored at n={cur['n']}, candidate at n={a.n}. "
              f"Re-score one of them under the other's protocol, or pass --force with a reason.")
        sys.exit(1)
    if a.score <= cur["score"] and not a.force:
        print(f"REFUSED: incumbent {cur['card']} scored {cur['score']} (n={cur['n']}) >= {a.score}")
        sys.exit(1)
    print(f"replacing incumbent {cur['card']} @ {cur['score']} (n={cur['n']})")

need = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
missing = [f for f in need if not os.path.exists(os.path.join(a.src, f))]
if missing:
    sys.exit(f"REFUSED: {a.src} is missing {missing}")
if not any(f.endswith(".safetensors") for f in os.listdir(a.src)):
    sys.exit(f"REFUSED: {a.src} has no safetensors")

tmp = FINAL + ".new"
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(a.src, tmp)
json.dump(
    {"score": a.score, "n": a.n, "card": a.card, "src": a.src, "eval_path": a.eval_path},
    open(os.path.join(tmp, "PROMOTED.json"), "w"),
    indent=2,
)
old = FINAL + ".old"
shutil.rmtree(old, ignore_errors=True)
if os.path.exists(FINAL):
    os.rename(FINAL, old)
os.rename(tmp, FINAL)
shutil.rmtree(old, ignore_errors=True)
print(f"promoted {a.src} -> {FINAL} at {a.score} (n={a.n}, {a.card})")
print(sorted(os.listdir(FINAL)))
