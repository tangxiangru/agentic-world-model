#!/usr/bin/env python3
"""Auxiliary diversity rows: OpenMathInstruct-2's augmented_math slice, kept
only where the expected answer is a plain number so the target can end in the
same "ANSWER: <n>" line the grader reads.  One solution per distinct problem.
"""
import glob, json, os, random, re, sys
import pyarrow.parquet as pq
sys.path.insert(0, "/home/ben/task/scripts")
from build_data import clean_solution

TASK = "/home/ben/task"
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
EOT = "<end_of_turn>"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 38000

tpl = json.load(open(f"{TASK}/data/eval_prompt.json"))["prompt_template"]
seen, rows = set(), []
files = sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                         "snapshots/*/data/train_1M-*.parquet"))
for f in files:
    t = pq.read_table(f, columns=["problem", "generated_solution",
                                  "expected_answer", "problem_source"])
    for s, p, so, an in zip(t.column("problem_source").to_pylist(),
                            t.column("problem").to_pylist(),
                            t.column("generated_solution").to_pylist(),
                            t.column("expected_answer").to_pylist()):
        if s not in ("math", "augmented_math"):
            continue
        an = (an or "").strip()
        if not NUMERIC.match(an):
            continue
        p = p.strip()
        if p in seen:
            continue
        body = clean_solution(so, an)
        if body is None or "ANSWER:" in so:
            continue
        seen.add(p)
        rows.append({"question": p, "prompt": tpl.replace("{prompt}", p),
                     "target": body + EOT, "answer": an, "src": f"omi2:{s}"})
    print(f"  {os.path.basename(f)}: {len(rows)}", flush=True)

random.Random(0).shuffle(rows)
rows = rows[:N]
with open(f"{TASK}/data/math_aux.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} math-diversity rows")
