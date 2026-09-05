"""Emit the rows of sft_v1.jsonl that run 1 did not consume (same shuffle, seed 0)."""
import json
import random
import sys

src, dst, n_used = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(src)]
random.seed(0)
random.shuffle(rows)
rest = rows[n_used:]
with open(dst, "w") as f:
    for r in rest:
        f.write(json.dumps(r) + "\n")
print(f"{len(rows)} total -> {len(rest)} unused written to {dst}")
