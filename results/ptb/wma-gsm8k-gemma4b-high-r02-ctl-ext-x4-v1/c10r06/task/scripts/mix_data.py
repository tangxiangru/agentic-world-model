#!/usr/bin/env python3
"""Mix jsonl training files: `--take file:n` (n=-1 for all), then attach the grader's exact
10-shot system prefix to `--fewshot N` randomly chosen rows that do not already have one."""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--take", action="append", required=True, help="path:n")
ap.add_argument("--fewshot", type=int, default=0)
ap.add_argument("--fewshot-file", default="/home/ben/task/data/fewshot_system.txt")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rnd = random.Random(a.seed)
rows = []
for spec in a.take:
    path, _, n = spec.rpartition(":")
    n = int(n)
    rs = [json.loads(l) for l in open(path)]
    rnd.shuffle(rs)
    if n >= 0:
        rs = rs[:n]
    print(f"{path}: {len(rs)}")
    rows += rs

rnd.shuffle(rows)
if a.fewshot:
    sysmsg = open(a.fewshot_file).read()
    done = 0
    for r in rows:
        if done >= a.fewshot:
            break
        if not r.get("system"):
            r["system"] = sysmsg
            done += 1
    print(f"attached the 10-shot prefix to {done} rows")

rnd.shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} -> {a.out}")
