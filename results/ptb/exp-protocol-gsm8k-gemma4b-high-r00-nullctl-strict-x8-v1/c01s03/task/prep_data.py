#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 + GSM8K train."""
import argparse, glob, json, random, re, collections
import pyarrow.parquet as pq

PROMPT = (
    'Solve the following math problem step by step. The last line of your response should be of the '
    'form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    '{prompt}\n\n'
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without '
    'quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed '
    'command.\n\nReasoning:'
)

BOXED_RE = re.compile(r'\\boxed\{')


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        out.append(text[m.end():j - 1])
        i = j
    return ''.join(out)


NUM_RE = re.compile(r'-?\d[\d,]*\.?\d*')


def is_plain_number(s: str) -> bool:
    s = s.strip().replace(',', '')
    if s.startswith('-'):
        s = s[1:]
    return bool(re.fullmatch(r'\d+(\.\d+)?', s))


def norm_num(s: str) -> str:
    s = s.strip().replace(',', '').replace('$', '')
    try:
        f = float(s)
    except ValueError:
        return s
    if f == int(f):
        return str(int(f))
    return ('%.6f' % f).rstrip('0').rstrip('.')


def build_solution(sol: str, ans: str) -> str | None:
    sol = strip_boxed(sol).strip()
    # kill leftover latex display noise that hurts a plain-number benchmark
    if '\\[' in sol or '\\begin{' in sol:
        return None
    sol = re.sub(r'\n{3,}', '\n\n', sol)
    a = norm_num(ans)
    if not sol:
        return None
    return sol + f'\n\nANSWER: {a}'


def gsm8k_reference(rec):
    q = rec['question'].strip()
    a = rec['answer']
    body, _, final = a.rpartition('####')
    body = re.sub(r'<<[^>]*>>', '', body).strip()
    final = norm_num(final.strip())
    if not body:
        return None
    return q, body + f'\n\nANSWER: {final}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='work/sft1.jsonl')
    ap.add_argument('--max-per-problem', type=int, default=2)
    ap.add_argument('--n-gsm', type=int, default=70000)
    ap.add_argument('--n-math', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/'
                             'snapshots/*/data/*.parquet'))
    by_problem_gsm = collections.defaultdict(list)
    by_problem_math = collections.defaultdict(list)
    for f in files:
        t = pq.read_table(f, columns=['problem', 'generated_solution', 'expected_answer',
                                      'problem_source'])
        for prob, sol, ans, src in zip(t.column('problem').to_pylist(),
                                       t.column('generated_solution').to_pylist(),
                                       t.column('expected_answer').to_pylist(),
                                       t.column('problem_source').to_pylist()):
            if not is_plain_number(ans):
                continue
            if len(sol) > 3500:
                continue
            d = by_problem_gsm if 'gsm8k' in src else by_problem_math
            if len(d[prob]) < args.max_per_problem * 2:
                d[prob].append((sol, ans))
        print('scanned', f, len(by_problem_gsm), len(by_problem_math), flush=True)

    def collect(d, n_target, tag):
        keys = list(d.keys())
        rng.shuffle(keys)
        rows, k = [], 0
        while len(rows) < n_target and k < args.max_per_problem:
            for prob in keys:
                if k < len(d[prob]):
                    sol, ans = d[prob][k]
                    s = build_solution(sol, ans)
                    if s:
                        rows.append({'question': prob.strip(), 'solution': s, 'answer': norm_num(ans), 'src': tag})
                    if len(rows) >= n_target:
                        break
            k += 1
        return rows

    rows = collect(by_problem_gsm, args.n_gsm, 'omi_gsm')
    print('gsm rows', len(rows))
    mrows = collect(by_problem_math, args.n_math, 'omi_math')
    print('math rows', len(mrows))
    rows += mrows

    # original gsm8k train reference solutions (terse, human-written)
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main')['train']
    nref = 0
    for rec in ds:
        r = gsm8k_reference(rec)
        if r:
            rows.append({'question': r[0], 'solution': r[1], 'answer': r[1].split('ANSWER: ')[-1], 'src': 'gsm8k_ref'})
            nref += 1
    print('gsm8k reference rows', nref)

    rng.shuffle(rows)
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print('wrote', len(rows), '->', args.out)


if __name__ == '__main__':
    main()
