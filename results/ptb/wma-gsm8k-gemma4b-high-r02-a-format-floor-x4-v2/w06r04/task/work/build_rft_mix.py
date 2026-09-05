"""Mix the rejection-sampled rows with a replay slice of the SFT file.

Replay matters because the RFT rows are all on-policy and all solvable-by-the
current-model; training on them alone narrows the distribution the SFT run
established. The replay slice is drawn from the same file exp-02 trained on.
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--rft", required=True)
ap.add_argument("--sft", required=True)
ap.add_argument("--replay", type=int, default=20000)
ap.add_argument("--out", required=True)
ap.add_argument("--min-completion-chars", type=int, default=0,
                help="drop RFT rows whose target is shorter than this; the corpus must not "
                     "pull the model's reasoning length below where the parent already is")
ap.add_argument("--seed", type=int, default=2)
a = ap.parse_args()

rft = [json.loads(l) for l in open(a.rft)]
n_before = len(rft)
if a.min_completion_chars:
    rft = [r for r in rft if len(r["completion"]) >= a.min_completion_chars]
    print(f"dropped {n_before - len(rft)} rft rows shorter than {a.min_completion_chars} chars")
sft = [json.loads(l) for l in open(a.sft)]
rng = random.Random(a.seed)
rng.shuffle(sft)
rows = rft + sft[: a.replay]
rng.shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows ({len(rft)} rft + {min(a.replay, len(sft))} replay) to {a.out}")
