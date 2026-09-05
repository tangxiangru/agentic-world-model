#!/usr/bin/env python3
"""Concatenate pre-tokenized corpora, optionally taking only the first N rows of
each, into one .npz the trainer can read.

usage: python merge_npz.py --out data/mix.npz a.npz:20000 b.npz:all
"""
from __future__ import annotations

import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("parts", nargs="+", help="path.npz[:n|:all]")
args = ap.parse_args()

rows, plens = [], []
for part in args.parts:
    path, _, spec = part.partition(":")
    z = np.load(path)
    flat, offs, pl = z["flat"], z["offsets"], z["prompt_lens"]
    n = len(pl) if spec in ("", "all") else min(int(spec), len(pl))
    idx = np.random.default_rng(args.seed).permutation(len(pl))[:n]
    for i in idx:
        rows.append(flat[offs[i]:offs[i + 1]])
        plens.append(int(pl[i]))
    print(f"{path}: took {n} of {len(pl)}")

order = np.random.default_rng(args.seed + 1).permutation(len(rows))
rows = [rows[i] for i in order]
plens = [plens[i] for i in order]

lens = [len(r) for r in rows]
offs = np.zeros(len(rows) + 1, dtype=np.int64)
offs[1:] = np.cumsum(lens)
np.savez(args.out, flat=np.concatenate(rows), offsets=offs,
         prompt_lens=np.array(plens, dtype=np.int32))
print(f"wrote {args.out}: {len(rows)} rows, {offs[-1]/1e6:.1f}M tokens, "
      f"max len {max(lens)}")
