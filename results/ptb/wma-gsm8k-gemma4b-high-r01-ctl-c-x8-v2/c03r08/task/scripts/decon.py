#!/usr/bin/env python3
"""Run the provided contamination checker over an SFT jsonl; non-zero exit on any hit."""
import argparse, json, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

tmp = Path(f"/home/ben/task/data/decon_input_{a.tag}.jsonl")
with tmp.open("w") as out:
    for i, l in enumerate(Path(a.input).open()):
        r = json.loads(l)
        out.write(json.dumps({"id": i, "text": r["question"] + "\n" + r["completion"]}) + "\n")
err = Path(f"/home/ben/task/logs/decon_{a.tag}.err")
res = subprocess.run(
    [sys.executable, "/home/ben/contamination_check.py", "--reference",
     "/home/ben/test_data.json", "--input", str(tmp)],
    stdout=open(f"/home/ben/task/logs/decon_{a.tag}.jsonl", "w"),
    stderr=open(err, "w"))
print(err.read_text()[-400:])
sys.exit(res.returncode)
