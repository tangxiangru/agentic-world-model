#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data generation: sample solutions from the current
model on TRAIN-split problems, keep the ones that reach the known correct answer."""
from __future__ import annotations
import argparse, glob, json, os, random, re, sys
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_data import USER_TMPL, clean_sol, norm_answer, is_plain_number, build_fewshot_pool

RNG = random.Random(7)
BOS, SOT, EOT = '<bos>', '<start_of_turn>', '<end_of_turn>'
ANS_RE = re.compile(r'ANSWER:\s*(.+?)\s*$', re.MULTILINE)


def extract(text):
    m = ANS_RE.findall(text)
    if not m:
        return None
    return norm_answer(m[-1])


def same(a, b):
    if a is None:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def load_problems(n_aug):
    f = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet'))[0]
    probs = []
    for r in pq.read_table(f).to_pylist():
        sol, ans = r['answer'].split('####')
        probs.append({'q': r['question'].strip(), 'a': norm_answer(ans), 'src': 'gsm8k_train'})
    if n_aug > 0:
        files = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet'))
        aug = {}
        for fp in files:
            t = pq.read_table(fp)
            for r in t.to_pylist():
                if r['problem_source'] != 'augmented_gsm8k':
                    continue
                if not is_plain_number(r['expected_answer']) or len(r['problem']) > 1500:
                    continue
                aug.setdefault(r['problem'].strip(), norm_answer(r['expected_answer']))
        aug = list(aug.items())
        RNG.shuffle(aug)
        for q, a in aug[:n_aug]:
            probs.append({'q': q, 'a': a, 'src': 'aug_gsm8k'})
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default='data/rft.jsonl')
    ap.add_argument('--n-aug', type=int, default=25000)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--keep-per-problem', type=int, default=2)
    ap.add_argument('--fewshot-frac', type=float, default=0.16)
    ap.add_argument('--gpu-util', type=float, default=0.85)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    probs = load_problems(args.n_aug)
    if args.limit:
        probs = probs[:args.limit]
    print('problems', len(probs), flush=True)

    prompts = []
    for p in probs:
        user = USER_TMPL.format(prompt=p['q'])
        prompts.append(f"{BOS}{SOT}user\n{user}{EOT}\n{SOT}model\n")

    llm = LLM(model=args.model, dtype='bfloat16', gpu_memory_utilization=args.gpu_util,
              max_model_len=1536, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=0)
    tokenized = [{'prompt_token_ids': tok(pr, add_special_tokens=False)['input_ids']} for pr in prompts]
    outs = llm.generate(tokenized, sp)

    fewshot_pool = build_fewshot_pool(None)
    n_kept = 0
    n_solved = 0
    stats = {}
    with open(args.out, 'w') as f:
        for p, o in zip(probs, outs):
            cands = []
            seen = set()
            for c in o.outputs:
                txt = c.text.strip()
                if not same(extract(txt), p['a']):
                    continue
                if '\\boxed' in txt or len(txt) < 20:
                    continue
                key = re.sub(r'\s+', ' ', txt.lower())
                if key in seen:
                    continue
                seen.add(key)
                cands.append(txt)
            if not cands:
                continue
            n_solved += 1
            cands.sort(key=len)          # prefer concise correct solutions
            for txt in cands[:args.keep_per_problem]:
                system = None
                if RNG.random() < args.fewshot_frac:
                    k = RNG.choice([1, 2, 3, 4, 10, 10])
                    system = '\n\n'.join(RNG.sample(fewshot_pool, k))
                f.write(json.dumps({
                    'system': system,
                    'user': USER_TMPL.format(prompt=p['q']),
                    'assistant': txt,
                    'source': 'rft_' + p['src'],
                }) + '\n')
                n_kept += 1
                stats[p['src']] = stats.get(p['src'], 0) + 1
    print(f'solved {n_solved}/{len(probs)} = {n_solved/len(probs):.3f}; kept {n_kept}', stats)


if __name__ == '__main__':
    main()
