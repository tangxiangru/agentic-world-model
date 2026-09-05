"""Second sampling pool: questions from sft_v1/sft_v2 that the round-1 pool never
covered, so round 2 mines failures on problems the model has not been sampled on."""
import json, random
from build_sft_data import norm_q

seen = set()
for f in ("probe250", "rft_pool"):
    for l in open(f"/home/ben/task/data/{f}.jsonl"):
        seen.add(norm_q(json.loads(l)["question"]))
rows, taken = [], set()
for f in ("sft_v1", "sft_v2"):
    for l in open(f"/home/ben/task/data/{f}.jsonl"):
        r = json.loads(l)
        q = norm_q(r["question"])
        if q in seen or q in taken:
            continue
        taken.add(q)
        rows.append({"id": f"p2-{len(rows)}", "question": r["question"], "gold": r["answer"], "src": f})
random.Random(5).shuffle(rows)
rows = rows[:22000]
with open("/home/ben/task/data/rft_pool2.jsonl", "w") as o:
    for r in rows:
        o.write(json.dumps(r) + "\n")
print("wrote rft_pool2.jsonl:", len(rows))
