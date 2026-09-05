"""Tokenize + pack a prompt/completion jsonl into fixed-length training blocks.

Kept out of the trainer so the (slow, CPU-only) tokenize step can be cached and
reused across runs. Output is a single .npz:
  ids   int32 [n_blocks, pack_len]
  lab   int32 [n_blocks, pack_len]   (-100 where no loss)
  pos   int32 [n_blocks, pack_len]   (restarts at 0 on every example boundary)

batch_size 1 over these blocks + these position_ids is the shape
transformers' `_is_packed_sequence` recognises, so flash-attention-2 runs
varlen and there is no cross-example attention.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
PAD_ID = 0


def tokenize(path, tok, max_row_len, limit=None, chunk=2000):
    prompts, comps = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            prompts.append(r["prompt"])
            comps.append(r["completion"])
    rows = []
    over = 0
    for s in range(0, len(prompts), chunk):
        P = tok(prompts[s:s + chunk], add_special_tokens=False)["input_ids"]
        C = tok(comps[s:s + chunk], add_special_tokens=False)["input_ids"]
        for p, c in zip(P, C):
            n = len(p) + len(c)
            if n > max_row_len:
                over += 1
                continue
            rows.append((np.array(p + c, dtype=np.int32), len(p)))
        if (s // chunk) % 20 == 0:
            print(f"  tokenized {s + len(P)}/{len(prompts)}", flush=True)
    n0 = len(prompts)
    print(f"[tok] rows={n0} kept={len(rows)} over_{max_row_len}={over} ({over / n0:.3%})", flush=True)
    assert over / n0 < 0.02, "more than 2% of rows exceed max_row_len"
    return rows


def pack(rows, pack_len, seed=0, lookahead=512):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    lens = np.array([len(rows[i][0]) for i in order], dtype=np.int32)
    used = np.zeros(len(order), dtype=bool)

    blocks = []
    cur, room = [], pack_len
    i = 0
    total = 0
    while i < len(order):
        if used[i]:
            i += 1
            continue
        j = i
        if lens[j] > room:
            j = -1
            hi = min(i + lookahead, len(order))
            cand = np.nonzero((~used[i:hi]) & (lens[i:hi] <= room))[0]
            if cand.size:
                j = i + int(cand[0])
            else:
                blocks.append(cur)
                total += pack_len - room
                cur, room = [], pack_len
                continue
        used[j] = True
        cur.append(order[j])
        room -= int(lens[j])
        if j == i:
            i += 1
    if cur:
        blocks.append(cur)
        total += pack_len - room

    nb = len(blocks)
    ids = np.zeros((nb, pack_len), dtype=np.int32)
    lab = np.full((nb, pack_len), -100, dtype=np.int32)
    pos = np.zeros((nb, pack_len), dtype=np.int32)
    for b, idxs in enumerate(blocks):
        off = 0
        for k in idxs:
            arr, plen = rows[k]
            n = len(arr)
            ids[b, off:off + n] = arr
            lab[b, off + plen:off + n] = arr[plen:]
            pos[b, off:off + n] = np.arange(n, dtype=np.int32)
            off += n
        if off < pack_len:
            pos[b, off:] = np.arange(pack_len - off, dtype=np.int32)
    fill = total / (nb * pack_len)
    ntok = int((lab != -100).sum())
    print(f"[pack] blocks={nb} pack_len={pack_len} fill={fill:.4f} loss_tokens={ntok/1e6:.1f}M "
          f"total_tokens={nb * pack_len / 1e6:.1f}M", flush=True)
    assert fill > 0.90, f"packing efficiency {fill:.3f} too low"
    return ids, lab, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-row-len", type=int, default=2048)
    ap.add_argument("--pack-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    rows = tokenize(args.data, tok, args.max_row_len, args.limit)
    ids, lab, pos = pack(rows, args.pack_len, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, ids=ids, lab=lab, pos=pos)
    print("[done] wrote", args.out, ids.shape, flush=True)


if __name__ == "__main__":
    main()
