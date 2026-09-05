"""Second, larger pool of grade-school CoT rows from the cached OpenMathInstruct-2 5M split."""
import glob
import json
import random
import sys

import pyarrow.parquet as pq

from common import clean_latex, norm_num, strip_boxed

random.seed(7)
OUT = sys.argv[1]
WANT = int(sys.argv[2])

# Questions already represented in v1 (avoid re-showing identical problems).
seen_q = set()
seen_pair = set()
for path in ("data/sft_v1.jsonl",):
    for line in open(path):
        r = json.loads(line)
        seen_pair.add((r["question"], r["solution"]))

files = sorted(glob.glob(
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/**/train_5M-*.parquet",
    recursive=True))
print(len(files), "parquet files", file=sys.stderr)

rows = []
for fp in files:
    if len(rows) >= WANT:
        break
    tbl = pq.read_table(fp, columns=["problem", "generated_solution",
                                     "expected_answer", "problem_source"])
    src = tbl.column("problem_source").to_pylist()
    keep = [i for i, s in enumerate(src) if s in ("gsm8k", "augmented_gsm8k")]
    if not keep:
        continue
    probs = tbl.column("problem").to_pylist()
    sols = tbl.column("generated_solution").to_pylist()
    answ = tbl.column("expected_answer").to_pylist()
    random.shuffle(keep)
    for i in keep:
        if len(rows) >= WANT:
            break
        ans = norm_num(answ[i])
        if ans is None:
            continue
        q = probs[i].strip()
        if len(q) < 20:
            continue
        body = clean_latex(strip_boxed(sols[i])).strip()
        if not body or len(body) > 4000 or "\\" in body or "boxed" in body:
            continue
        body = body + "\n\nANSWER: " + ans
        if (q, body) in seen_pair:
            continue
        seen_pair.add((q, body))
        rows.append({"question": q, "solution": body, "answer": ans,
                     "source": src[i] + "_v2"})
    print(f"  {fp.split('/')[-1]}: total {len(rows)}", file=sys.stderr)

random.shuffle(rows)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} -> {OUT}", file=sys.stderr)
