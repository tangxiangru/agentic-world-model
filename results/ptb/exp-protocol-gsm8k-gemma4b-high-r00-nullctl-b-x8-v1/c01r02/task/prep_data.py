#!/usr/bin/env python3
"""Build SFT training data for GSM8K from train-split-derived math corpora."""
from __future__ import annotations
import glob, json, random, re, argparse, os
import pyarrow.parquet as pq

RNG = random.Random(1234)

USER_TMPL = (
    'Solve the following math problem step by step. The last line of your response '
    'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    '{prompt}\n\n'
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" '
    '(without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n'
    'Reasoning:'
)

BOXED_RE = re.compile(r'\\boxed\s*\{')


def strip_boxed(s: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(s) and depth:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
            j += 1
        out.append(s[m.end():j - 1])
        i = j
    return ''.join(out)


NUM_RE = re.compile(r'^-?\$?\d[\d,]*(\.\d+)?%?$')


def norm_answer(a: str) -> str:
    a = a.strip().replace('$', '').replace(',', '').rstrip('.')
    return a


def is_plain_number(a: str) -> bool:
    a = norm_answer(a)
    try:
        float(a)
        return True
    except Exception:
        return False


def clean_sol(sol: str) -> str:
    sol = strip_boxed(sol).strip()
    # drop calculator annotations from raw gsm8k
    sol = re.sub(r'<<[^>]*>>', '', sol)
    sol = re.sub(r'\n{3,}', '\n\n', sol)
    return sol.strip()


def make_record(question: str, solution: str, answer: str, src: str):
    return {
        'question': question.strip(),
        'solution': solution,
        'answer': answer,
        'source': src,
    }


def load_gsm8k_train():
    f = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet'))[0]
    t = pq.read_table(f).to_pylist()
    out = []
    for r in t:
        q = r['question']
        a = r['answer']
        sol, ans = a.split('####')
        out.append(make_record(q, clean_sol(sol), norm_answer(ans), 'gsm8k_train'))
    return out


def load_omi2():
    files = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet'))
    buckets = {'gsm8k': [], 'augmented_gsm8k': [], 'math': [], 'augmented_math': []}
    for f in files:
        t = pq.read_table(f)
        for r in t.to_pylist():
            src = r['problem_source']
            if src not in buckets:
                continue
            ans = r['expected_answer']
            sol = r['generated_solution']
            if not ans or not sol:
                continue
            if src.endswith('gsm8k') and not is_plain_number(ans):
                continue
            if len(sol) > 4000 or len(r['problem']) > 2000:
                continue
            buckets[src].append(make_record(r['problem'], clean_sol(sol), norm_answer(ans) if is_plain_number(ans) else ans.strip(), src))
    return buckets


def load_orca():
    f = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--microsoft--orca-math-word-problems-200k/snapshots/*/data/*.parquet'))[0]
    out = []
    for r in pq.read_table(f).to_pylist():
        q, a = r['question'], r['answer']
        if not q or not a or len(a) > 4000 or len(q) > 2000:
            continue
        a = clean_sol(a)
        # final answer = last number in the solution
        nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', a.replace('\\', ' '))
        if not nums:
            continue
        ans = norm_answer(nums[-1])
        out.append(make_record(q, a, ans, 'orca'))
    return out


def build_fewshot_pool(gsm_train):
    """Few-shot blocks in exactly the eval's format, built from GSM8K TRAIN only."""
    f = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet'))[0]
    t = pq.read_table(f).to_pylist()
    pool = []
    for r in t:
        sol, ans = r['answer'].split('####')
        pool.append(f"{r['question']}\n\nReasoning:\n{sol.strip()}\n\nANSWER: {ans.strip()}")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/sft.jsonl')
    ap.add_argument('--n-aug-gsm8k', type=int, default=90000)
    ap.add_argument('--n-aug-math', type=int, default=30000)
    ap.add_argument('--n-orca', type=int, default=20000)
    ap.add_argument('--gsm-repeat', type=int, default=2)
    ap.add_argument('--fewshot-frac', type=float, default=0.25)
    args = ap.parse_args()

    gsm = load_gsm8k_train()
    print('gsm8k train', len(gsm))
    b = load_omi2()
    for k, v in b.items():
        print('omi2', k, len(v))
    orca = load_orca()
    print('orca', len(orca))

    recs = []
    recs += gsm * args.gsm_repeat
    recs += b['gsm8k']
    RNG.shuffle(b['augmented_gsm8k'])
    recs += b['augmented_gsm8k'][:args.n_aug_gsm8k]
    recs += b['math']
    RNG.shuffle(b['augmented_math'])
    recs += b['augmented_math'][:args.n_aug_math]
    RNG.shuffle(orca)
    recs += orca[:args.n_orca]

    RNG.shuffle(recs)
    fewshot_pool = build_fewshot_pool(gsm)

    n_fs = 0
    with open(args.out, 'w') as f:
        for r in recs:
            user = USER_TMPL.format(prompt=r['question'])
            system = None
            if RNG.random() < args.fewshot_frac:
                k = RNG.choice([1, 2, 3, 4, 10, 10])
                system = '\n\n'.join(RNG.sample(fewshot_pool, k))
                n_fs += 1
            assistant = r['solution'].rstrip() + f"\n\nANSWER: {r['answer']}"
            f.write(json.dumps({
                'system': system,
                'user': user,
                'assistant': assistant,
                'source': r['source'],
            }) + '\n')
    print('wrote', len(recs), 'to', args.out, '| fewshot-prefixed:', n_fs)


if __name__ == '__main__':
    main()
