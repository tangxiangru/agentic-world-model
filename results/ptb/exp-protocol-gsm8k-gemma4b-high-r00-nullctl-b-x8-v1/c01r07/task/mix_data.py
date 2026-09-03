#!/usr/bin/env python3
"""Mix RFT (on-policy, verified) data with a replay slice of the seed SFT data."""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--rft", nargs="+", required=True)
ap.add_argument("--replay", default="data/sft.jsonl")
ap.add_argument("--n-replay", type=int, default=30000)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = random.Random(args.seed)
recs = []
seen = set()
for path in args.rft:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["question"], r["completion_text"])
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
print(f"rft records: {len(recs)}")

if args.n_replay > 0:
    replay = []
    with open(args.replay) as f:
        for line in f:
            replay.append(json.loads(line))
    rng.shuffle(replay)
    recs += replay[: args.n_replay]
    print(f"+ replay: {min(args.n_replay, len(replay))}")

rng.shuffle(recs)
with open(args.out, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(recs)} -> {args.out}")
