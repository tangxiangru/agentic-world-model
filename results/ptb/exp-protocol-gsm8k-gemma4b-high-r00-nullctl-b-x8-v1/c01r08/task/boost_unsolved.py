import glob, json, collections
import pyarrow.parquet as pq
from prep_data import unbox, clean_ans, is_num, make, load_gsm8k_train

uns = {}
for l in open('data/rft_r1.jsonl.unsolved'):
    r = json.loads(l); uns[r['question'].strip()] = clean_ans(r['answer'])
print('unsolved', len(uns))

out = []
per = collections.Counter()
files = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet'))
for f in files:
    t = pq.read_table(f)
    for p, s, a, src in zip(t.column('problem').to_pylist(), t.column('generated_solution').to_pylist(),
                            t.column('expected_answer').to_pylist(), t.column('problem_source').to_pylist()):
        q = p.strip()
        if q not in uns: continue
        a = clean_ans(a)
        if a != uns[q] or not is_num(a) or '\\boxed{' not in s: continue
        if len(s) > 2600 or len(s) < 40: continue
        if per[q] >= 3: continue
        body = unbox(s).strip()
        if '[asy]' in body or '\\begin{tabular}' in body: continue
        per[q] += 1
        out.append(make(p, body, a))
# human gsm8k solutions for unsolved train questions
for ex in load_gsm8k_train():
    if ex['question'].strip() in uns:
        out.append(ex); out.append(ex)
print('boost rows', len(out), 'covered', len(per))
with open('data/unsolved_boost.jsonl','w') as f:
    for r in out: f.write(json.dumps(r)+'\n')
