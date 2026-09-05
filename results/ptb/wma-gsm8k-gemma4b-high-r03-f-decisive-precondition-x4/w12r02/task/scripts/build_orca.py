#!/usr/bin/env python3
"""Second-stage data with NEW problems: microsoft/orca-math-word-problems-200k
reformatted into the grader's target shape, mixed with fresh OpenMathInstruct-2
teacher solutions as a distribution anchor."""
import argparse, json, random, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render
from build_data import strip_boxed
from eval_local import last_number

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--n-orca', type=int, default=60000)
ap.add_argument('--n-teacher', type=int, default=60000)
ap.add_argument('--teacher-file', default='data/sft_v2_moresols.jsonl')
ap.add_argument('--fewshot-frac', type=float, default=0.10)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
import datasets
d = datasets.load_dataset('microsoft/orca-math-word-problems-200k', split='train')
tr = datasets.load_dataset('openai/gsm8k', 'main', split='train')

def fewshots(k):
    bl = []
    for i in rng.sample(range(len(tr)), k):
        r = tr[i]; parts = r['answer'].split('####')
        bl.append(render.fewshot_block(r['question'], '####'.join(parts[:-1]).strip(), parts[-1].strip()))
    return bl

kept, drop = [], 0
for r in d:
    q, ans = r['question'], r['answer']
    body = strip_boxed(ans).strip()
    if 'ANSWER' in body or '\\boxed' in body: drop += 1; continue
    if not (50 <= len(body) <= 2500): drop += 1; continue
    g = last_number(body)
    if g is None: drop += 1; continue
    # the target must not end on a different number than the declared answer
    if len(g) > 12: drop += 1; continue
    # the gold must be stated in the closing lines of the chain, otherwise the
    # ANSWER line is not supported by the reasoning the model is trained on
    if g not in body[-160:]: drop += 1; continue
    kept.append((q, body, g))
print(f'[orca] kept {len(kept)} dropped {drop}', flush=True)
rng.shuffle(kept); kept = kept[:a.n_orca]

teacher = [json.loads(l) for l in open(a.teacher_file)]
rng.shuffle(teacher); teacher = teacher[:a.n_teacher]

rows = []
for q, body, g in kept:
    k = rng.choice([1,2,3,4,10]) if rng.random() < a.fewshot_frac else 0
    rows.append({'prompt': render.build_prompt(q, fewshots(k) if k else None),
                 'completion': render.build_completion(body + f"\n\nANSWER: {g}"),
                 'answer': g, 'source': 'orca-math', 'nshot': k})
rows += teacher
rng.shuffle(rows)
with open(a.out, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print(f'[orca] wrote {len(rows)} -> {a.out}', flush=True)
