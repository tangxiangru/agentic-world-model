"""Fresh gsm8k-style rows from OpenMathInstruct-2 train_2M that no earlier card used."""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TASK_DIR, read_jsonl, write_jsonl, user_prompt
from build_sft import make_row, norm_answer, is_int_answer
from datasets import load_dataset

rng = random.Random(7)
dev_qs = set(json.load(open(os.path.join(TASK_DIR, "data", "dev500_questions.json"))))
used = set()
for f in ["sft_v1.jsonl", "sft_v2.jsonl"]:
    for r in read_jsonl(os.path.join(TASK_DIR, "data", f)):
        used.add((r["question"], r["completion"]))
print("rows already used:", len(used))

omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_2M")
srcs = omi["problem_source"]
cand = [i for i, s in enumerate(srcs) if s in ("gsm8k", "augmented_gsm8k")]
rng.shuffle(cand)
print("gsm8k-sourced rows in train_2M:", len(cand))

out, per_problem = [], {}
for i in cand:
    if len(out) >= 24000:
        break
    r = omi[i]
    q = r["problem"]
    if q in dev_qs or per_problem.get(q, 0) >= 2:
        continue
    a = norm_answer(r["expected_answer"])
    if not is_int_answer(a):
        continue
    sol = r["generated_solution"]
    if len(sol) > 4000:
        continue
    row = make_row(q, sol, a, "omi2_" + r["problem_source"])
    if row is None or (row["question"], row["completion"]) in used:
        continue
    per_problem[q] = per_problem.get(q, 0) + 1
    out.append(row)
write_jsonl(os.path.join(TASK_DIR, "data", "sft_v3.jsonl"), out)
print("wrote", len(out), "fresh rows")
