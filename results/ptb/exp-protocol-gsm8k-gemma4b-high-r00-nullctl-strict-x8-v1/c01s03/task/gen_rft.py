#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per problem with vLLM,
keep the ones whose final ANSWER matches the reference answer."""
import argparse, glob, json, os, random, re, collections
import pyarrow.parquet as pq

PROMPT = (
    'Solve the following math problem step by step. The last line of your response should be of the '
    'form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    '{prompt}\n\n'
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without '
    'quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed '
    'command.\n\nReasoning:'
)


def norm_num(s):
    s = str(s).strip().replace(',', '').replace('$', '').rstrip('.')
    try:
        f = float(s)
    except ValueError:
        return None
    return round(f, 4)


ANS_RE = re.compile(r'ANSWER:\s*\$?(-?[\d,]*\.?\d+)')


def extract(text):
    m = ANS_RE.findall(text)
    if not m:
        return None
    return norm_num(m[-1])


def load_problems(n_aug, seed):
    from datasets import load_dataset
    rng = random.Random(seed)
    probs = []
    ds = load_dataset('openai/gsm8k', 'main')['train']
    for r in ds:
        probs.append({'question': r['question'].strip(),
                      'answer': r['answer'].rpartition('####')[2].strip(), 'src': 'gsm8k'})
    aug = {}
    files = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/'
                             'snapshots/*/data/*.parquet'))
    for f in files:
        t = pq.read_table(f, columns=['problem', 'expected_answer', 'problem_source'])
        for p, a, s in zip(t.column('problem').to_pylist(),
                           t.column('expected_answer').to_pylist(),
                           t.column('problem_source').to_pylist()):
            if s == 'augmented_gsm8k' and norm_num(a) is not None:
                aug.setdefault(p.strip(), a.strip())
    keys = list(aug.keys())
    rng.shuffle(keys)
    for k in keys[:n_aug]:
        probs.append({'question': k, 'answer': aug[k], 'src': 'aug'})
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='work/sft1_model')
    ap.add_argument('--out', default='work/rft.jsonl')
    ap.add_argument('--n-aug', type=int, default=24000)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=640)
    ap.add_argument('--keep-per-problem', type=int, default=2)
    ap.add_argument('--gpu-util', type=float, default=0.85)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    probs = load_problems(args.n_aug, args.seed)
    print('problems', len(probs), flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    with open('templates/gemma3.jinja') as f:
        tok.chat_template = f.read()

    prompts = [tok.apply_chat_template(
        [{'role': 'user', 'content': PROMPT.format(prompt=p['question'])}],
        tokenize=False, add_generation_prompt=True) for p in probs]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=2048,
              dtype='bfloat16', enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop=['<end_of_turn>'])
    outs = llm.generate(prompts, sp)

    rows, stats = [], collections.Counter()
    solved_hist = collections.Counter()
    for p, o in zip(probs, outs):
        gold = norm_num(p['answer'])
        good, seen = [], set()
        for c in o.outputs:
            txt = c.text.strip()
            if extract(txt) != gold or gold is None:
                continue
            if not txt.rstrip().endswith(f'ANSWER: {p["answer"]}'.rstrip()):
                # normalise trailing formatting so the target always ends at the answer
                idx = txt.rfind('ANSWER:')
                if idx < 0:
                    continue
                txt = txt[:idx] + f'ANSWER: {p["answer"]}'
            key = re.sub(r'\s+', ' ', txt)[:200]
            if key in seen:
                continue
            seen.add(key)
            good.append(txt)
        solved_hist[len(good)] += 1
        good.sort(key=len)
        for txt in good[:args.keep_per_problem]:
            rows.append({'question': p['question'], 'solution': txt,
                         'answer': p['answer'], 'src': p['src'],
                         'n_correct': len(good), 'k': args.k})
        stats[p['src'] + ('_solved' if good else '_unsolved')] += 1

    print('stats', dict(stats))
    print('solved histogram', dict(sorted(solved_hist.items())))
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print('wrote', len(rows), '->', args.out)


if __name__ == '__main__':
    main()
