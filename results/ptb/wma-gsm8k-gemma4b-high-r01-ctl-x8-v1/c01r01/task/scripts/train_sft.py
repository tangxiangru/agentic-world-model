#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on prompt/completion jsonl.

Design notes (why this is not TRL):
  * The grader renders prompts with templates/gemma3.jinja; build_data.py has
    already rendered them, so the trainer must tokenize with
    add_special_tokens=False and append exactly one <end_of_turn>.
  * vocab is 262144, so materialising [B, T, V] fp32 logits is the memory wall.
    Loss is computed on the selected label positions only, in checkpointed
    chunks, so peak logit memory is chunk*V*4 bytes regardless of batch size.
  * fp32 master weights + bnb AdamW8bit fits an H100 with room to spare and
    avoids pure-bf16 update rounding.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, "/home/ben/task/scripts")
import fmt  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenize_all(path, tok, max_seq_len, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        log(f"loaded tokenised cache {cache}")
        return list(d["ids"]), list(d["nprompt"])
    ids_all, nprompt_all = [], []
    prompts, comps = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["prompt"])
            c = r["completion"]
            assert c.endswith(fmt.STOP_TOKEN), "target does not end with the grader's stop token"
            comps.append(c)
    log(f"tokenising {len(prompts)} rows")
    B = 2000
    n_trunc = 0
    for i in range(0, len(prompts), B):
        pe = tok(prompts[i:i + B], add_special_tokens=False)["input_ids"]
        ce = tok(comps[i:i + B], add_special_tokens=False)["input_ids"]
        for p, c in zip(pe, ce):
            if len(p) + len(c) > max_seq_len:
                n_trunc += 1
                continue
            ids_all.append(np.array(p + c, dtype=np.int32))
            nprompt_all.append(len(p))
    log(f"kept {len(ids_all)} rows, dropped {n_trunc} over max_seq_len={max_seq_len}")
    if cache:
        np.savez(cache, ids=np.array(ids_all, dtype=object), nprompt=np.array(nprompt_all))
    return ids_all, nprompt_all


def make_batches(lengths, token_budget, max_bs, rng):
    """Length-grouped, token-budget batches; batch order shuffled."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, curmax = [], [], 0
    for i in order:
        m = max(curmax, lengths[i])
        if cur and (m * (len(cur) + 1) > token_budget or len(cur) + 1 > max_bs):
            batches.append(cur)
            cur, curmax = [i], lengths[i]
        else:
            cur.append(i)
            curmax = m
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def chunked_ce_sum(hidden, lm_head, labels, chunk):
    """sum of CE over selected positions, logits recomputed per chunk."""
    w = lm_head.weight
    total = hidden.new_zeros((), dtype=torch.float32)

    def f(h, l):
        return F.cross_entropy(F.linear(h, w).float(), l, reduction="sum")

    n = hidden.shape[0]
    for i in range(0, n, chunk):
        h = hidden[i:i + chunk]
        l = labels[i:i + chunk]
        total = total + checkpoint(f, h, l, use_reentrant=False)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", default=fmt.BASE_SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--token-budget", type=int, default=12288)
    ap.add_argument("--max-bs", type=int, default=48)
    ap.add_argument("--accum", type=int, default=6)
    ap.add_argument("--ce-chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--save-every-frac", type=float, default=0.5,
                    help="save a checkpoint every this fraction of total training")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration
    import bitsandbytes as bnb

    tok = AutoTokenizer.from_pretrained(fmt.BASE_SNAPSHOT)
    ids_all, nprompt = tokenize_all(args.data, tok, args.max_seq_len, args.cache)
    lengths = [len(x) for x in ids_all]
    log(f"len p50={int(np.percentile(lengths,50))} p90={int(np.percentile(lengths,90))} "
        f"p99={int(np.percentile(lengths,99))} max={max(lengths)} total_tok={sum(lengths)/1e6:.1f}M")

    log(f"loading {args.init} in fp32")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation="sdpa")
    model.config.text_config.use_cache = False
    model.model.vision_tower.requires_grad_(False)
    model.model.multi_modal_projector.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.cuda()
    model.train()  # gradient checkpointing is a no-op unless the module is in training mode
    assert model.model.language_model.gradient_checkpointing, "grad checkpointing not on"
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params {sum(p.numel() for p in trainable)/1e9:.2f}B")

    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay, eps=1e-8)

    # --- schedule -----------------------------------------------------------
    epoch_batches = []
    n_ep = math.ceil(args.epochs)
    for e in range(n_ep):
        b = make_batches(lengths, args.token_budget, args.max_bs, random.Random(args.seed + e))
        epoch_batches.append(b)
    per_epoch_steps = len(epoch_batches[0]) // args.accum
    total_steps = int(per_epoch_steps * args.epochs)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    log(f"microbatches/epoch={len(epoch_batches[0])} accum={args.accum} total_steps={total_steps}")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        p = min(1.0, p)
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    pad_id = tok.pad_token_id or 0
    save_points = set()
    if args.save_every_frac and args.save_every_frac < 1.0:
        k = 1
        while k * args.save_every_frac < 1.0:
            save_points.add(int(total_steps * k * args.save_every_frac))
            k += 1

    def save(tag):
        d = os.path.join(args.out, tag)
        os.makedirs(d, exist_ok=True)
        log(f"saving {d}")
        model.config.text_config.use_cache = True
        m = model.to(torch.bfloat16)
        m.save_pretrained(d, safe_serialization=True)
        model.config.text_config.use_cache = False
        tok.save_pretrained(d)
        # exp-03: vLLM seeds its default SamplingParams from this file and the
        # grader never sets temperature, so the served model must ask for greedy.
        json.dump({"bos_token_id": 2, "cache_implementation": "hybrid",
                   "do_sample": False, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "temperature": 0.0, "top_k": 0, "top_p": 1.0,
                   "transformers_version": "4.57.3"},
                  open(os.path.join(d, "generation_config.json"), "w"), indent=2)
        try:
            AutoProcessor.from_pretrained(fmt.BASE_SNAPSHOT).save_pretrained(d)
        except Exception as e:
            log(f"processor save skipped: {e}")
        model.to(torch.float32)
        log(f"saved {d}")

    # --- train loop ---------------------------------------------------------
    step = 0
    t0 = time.time()
    losshist = []
    micro_iter = [i for e in range(n_ep) for i in [(e, b) for b in epoch_batches[e]]]
    done = False
    for gstart in range(0, len(micro_iter), args.accum):
        if done:
            break
        group = micro_iter[gstart:gstart + args.accum]
        if len(group) < args.accum:
            break
        # count label tokens in this optimiser window for exact normalisation
        ntok = sum(sum(lengths[i] - nprompt[i] for i in b) for _, b in group)
        loss_acc = 0.0
        for _, bidx in group:
            L = max(lengths[i] for i in bidx)
            inp = torch.full((len(bidx), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(bidx), L), dtype=torch.long)
            lab = torch.full((len(bidx), L), -100, dtype=torch.long)
            for r, i in enumerate(bidx):
                n = lengths[i]
                t = torch.from_numpy(ids_all[i].astype(np.int64))
                inp[r, :n] = t
                att[r, :n] = 1
                lab[r, nprompt[i]:n] = t[nprompt[i]:]
            inp, att, lab = inp.cuda(), att.cuda(), lab.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.model(input_ids=inp, attention_mask=att, return_dict=True)
                h = out.last_hidden_state[:, :-1, :]
                y = lab[:, 1:]
                sel = y != -100
                hs = h[sel]
                ys = y[sel]
                loss_sum = chunked_ce_sum(hs, model.lm_head, ys, args.ce_chunk)
            (loss_sum / ntok).backward()
            loss_acc += loss_sum.item()
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        gn = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        losshist.append(loss_acc / ntok)
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            log(f"step {step}/{total_steps} loss {np.mean(losshist[-10:]):.4f} lr {lr:.2e} "
                f"gnorm {gn:.2f} tok/s {sum(sum(lengths[i] for i in b) for _, b in micro_iter[:1])*0:.0f}"
                f" elapsed {el/60:.1f}m eta {el/step*(total_steps-step)/60:.1f}m "
                f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}G")
        if step in save_points:
            save(f"checkpoint-{step}")
        if step >= total_steps:
            done = True
    save("final")
    json.dump({"loss": losshist, "steps": step, "final_loss": float(np.mean(losshist[-20:]))},
              open(os.path.join(args.out, "trainlog.json"), "w"))
    log(f"done in {(time.time()-t0)/3600:.2f}h final_loss {np.mean(losshist[-20:]):.4f}")


if __name__ == "__main__":
    main()
