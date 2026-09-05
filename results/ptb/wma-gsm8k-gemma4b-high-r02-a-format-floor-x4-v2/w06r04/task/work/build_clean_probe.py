"""A dev set that is actually held out.

work/probe_set.jsonl turned out NOT to be: OpenMathInstruct-2's `gsm8k` slice
covers the whole GSM8K train split, so 194 of its 200 questions reached
sft_v2.jsonl. It is still legal (train items, never test) but it is no longer a
held-out measurement.

This builds 200 MetaMathQA GSM-derived problems whose normalised question text
appears in NO training file this session. Answers come from the dataset's
"The answer is: X" tail. GSM8K-train-derived, never test.
"""
import glob, json, random, re

def norm(q): return " ".join(q.split()).lower()

used = set()
for f in ["work/data/sft_v2.jsonl", "work/data/rft_mix_v1.jsonl",
          "work/data/rft_v1.jsonl", "work/data/rft_problems.jsonl"]:
    for l in open(f):
        used.add(norm(json.loads(l)["question"]))
print("distinct questions seen in training/sampling:", len(used))

NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
cand = []
for r in json.load(open(glob.glob("/home/ben/hf_cache/hub/datasets--meta-math--MetaMathQA/"
                                  "snapshots/*/MetaMathQA-395K.json")[0])):
    if r["type"] not in ("GSM_AnsAug", "GSM_Rephrased"):
        continue
    m = re.search(r"The answer is:\s*(.+?)\s*$", r["response"])
    if not m:
        continue
    a = m.group(1).strip().replace(",", "").replace("$", "")
    if not NUM.match(a):
        continue
    q = r["query"].strip()
    if norm(q) in used:
        continue
    used.add(norm(q))
    f = float(a)
    cand.append({"question": q, "gold": str(int(f)) if f == int(f) else str(f)})

random.Random(7).shuffle(cand)
rows = cand[:200]
with open("work/clean_probe.jsonl", "w") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"id": f"clean-{i}", **r}) + "\n")
print(f"wrote {len(rows)} clean probe rows from {len(cand)} candidates")
