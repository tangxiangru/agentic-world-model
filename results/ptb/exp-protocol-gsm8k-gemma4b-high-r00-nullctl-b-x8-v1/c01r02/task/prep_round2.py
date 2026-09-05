#!/usr/bin/env python3
"""Round-2 mixture: on-policy RFT data + fresh (unused) slices of the corpora."""
from __future__ import annotations
import argparse, json, random
from prep_data import (USER_TMPL, load_gsm8k_train, load_omi2, load_orca,
                       build_fewshot_pool)

RNG = random.Random(99)


def emit(recs, fewshot_pool, frac):
    out = []
    for r in recs:
        system = None
        if RNG.random() < frac:
            k = RNG.choice([1, 2, 3, 4, 10, 10])
            system = '\n\n'.join(RNG.sample(fewshot_pool, k))
        out.append({
            'system': system,
            'user': USER_TMPL.format(prompt=r['question']),
            'assistant': r['solution'].rstrip() + f"\n\nANSWER: {r['answer']}",
            'source': r['source'],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rft', default='data/rft.jsonl')
    ap.add_argument('--out', default='data/sft2.jsonl')
    ap.add_argument('--aug-gsm8k', type=int, default=30000)
    ap.add_argument('--aug-gsm8k-offset', type=int, default=58000)
    ap.add_argument('--orca', type=int, default=15000)
    ap.add_argument('--orca-offset', type=int, default=12000)
    ap.add_argument('--aug-math', type=int, default=6000)
    ap.add_argument('--aug-math-offset', type=int, default=12000)
    ap.add_argument('--gsm-repeat', type=int, default=1)
    ap.add_argument('--fewshot-frac', type=float, default=0.16)
    args = ap.parse_args()

    # NOTE: identical RNG seed/order as prep_data.py, so slicing by offset yields
    # samples that were NOT part of round 1.
    b = load_omi2()
    r1 = random.Random(1234)
    gsm = load_gsm8k_train()
    r1.shuffle(b['augmented_gsm8k'])
    orca = load_orca()
    r1.shuffle(b['augmented_math'])
    r1.shuffle(orca)

    recs = []
    recs += gsm * args.gsm_repeat
    recs += b['augmented_gsm8k'][args.aug_gsm8k_offset:args.aug_gsm8k_offset + args.aug_gsm8k]
    recs += orca[args.orca_offset:args.orca_offset + args.orca]
    recs += b['augmented_math'][args.aug_math_offset:args.aug_math_offset + args.aug_math]

    fewshot_pool = build_fewshot_pool(None)
    items = emit(recs, fewshot_pool, args.fewshot_frac)

    with open(args.rft) as f:
        rft = [json.loads(l) for l in f]
    items += rft
    RNG.shuffle(items)

    with open(args.out, 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    from collections import Counter
    print('wrote', len(items), Counter(i['source'] for i in items))


if __name__ == '__main__':
    main()
