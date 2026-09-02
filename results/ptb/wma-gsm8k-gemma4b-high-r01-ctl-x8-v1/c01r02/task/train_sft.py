#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style data.

Custom loop rather than HF Trainer for three reasons:
  * token-budgeted dynamic batching (length-bucketed) - the rows range from
    ~250 to ~2000 tokens because 30% carry the grader's 10-shot system message;
  * the lm_head is applied ONLY at supervised positions. gemma-3's vocab is
    262144, so a full-sequence float32 logit tensor is the memory bottleneck;
    completion-only loss labels ~30% of positions, so this cuts the peak by ~3x;
  * rows longer than --max-seq-len are DROPPED, never truncated (pitfall
    seq_len_truncation: truncating a completion-only row silently yields a row
    with zero loss tokens).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
# fragmentation was 16 GiB of reserved-but-unallocated at the exp-02 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--init", default=BASE)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-seq-len", type=int, default=2304)
    p.add_argument("--token-budget", type=int, default=16384, help="max padded tokens per micro-batch")
    p.add_argument("--micro-cap", type=int, default=32, help="max rows per micro-batch")
    p.add_argument("--label-budget", type=int, default=4096,
                   help="max SUPERVISED tokens per micro-batch. This is the real memory knob: the "
                        "float32 logit tensor is n_label x 262144, so 4096 labels ~= 4.3 GB of "
                        "logits + the same again for their gradient. Ignoring it OOMed exp-02 at "
                        "step 80 on a batch of 32 long-completion rows.")
    p.add_argument("--accum-rows", type=int, default=64, help="target rows per optimizer step")
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=0, help="optimizer steps between checkpoints (0=only final)")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--max-drop-frac", type=float, default=0.02)
    p.add_argument("--dry-run", action="store_true", help="tokenize + report, no GPU")
    return p.parse_args()


def load_rows(path, tok, max_seq_len, max_rows):
    rows, dropped, no_label = [], 0, 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pi = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            ci = tok(r["completion"], add_special_tokens=False)["input_ids"]
            if len(pi) + len(ci) > max_seq_len:
                dropped += 1
                continue
            if not ci:
                no_label += 1
                continue
            rows.append((pi, ci))
            if max_rows and len(rows) >= max_rows:
                break
    return rows, dropped, no_label


def make_batches(rows, token_budget, micro_cap, rng, label_budget):
    """Length-bucketed micro-batches under both a padded-token and a label-token budget."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    batches, cur, cur_max, cur_lab = [], [], 0, 0
    for i in order:
        L = len(rows[i][0]) + len(rows[i][1])
        nl = len(rows[i][1])
        m = max(cur_max, L)
        if cur and ((len(cur) + 1) * m > token_budget
                    or cur_lab + nl > label_budget
                    or len(cur) + 1 > micro_cap):
            batches.append(cur)
            cur, cur_max, cur_lab = [i], L, nl
        else:
            cur.append(i)
            cur_max, cur_lab = m, cur_lab + nl
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(rows, idxs, pad_id, device):
    seqs = [rows[i][0] + rows[i][1] for i in idxs]
    labs = [[-100] * len(rows[i][0]) + rows[i][1] for i in idxs]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    for j, (s, y) in enumerate(zip(seqs, labs)):
        ids[j, : len(s)] = torch.tensor(s)
        att[j, : len(s)] = 1
        lab[j, : len(y)] = torch.tensor(y)
    return ids.to(device), att.to(device), lab.to(device)


def get_text_model(model):
    """Return (callable returning last hidden state, lm_head)."""
    inner = model.model
    lm = inner.language_model if hasattr(inner, "language_model") else inner

    def fwd(input_ids, attention_mask):
        return lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)[0]

    return fwd, model.lm_head


def main():
    a = parse()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(BASE)
    rows, dropped, no_label = load_rows(a.data, tok, a.max_seq_len, a.max_rows)
    n_lab = sum(len(c) for _, c in rows)
    n_tot = sum(len(p) + len(c) for p, c in rows)
    print(f"[data] kept={len(rows)} dropped_too_long={dropped} ({dropped/max(1,len(rows)+dropped):.2%}) "
          f"empty_completion={no_label} tokens={n_tot/1e6:.1f}M label_tokens={n_lab/1e6:.1f}M", flush=True)
    assert dropped / max(1, len(rows) + dropped) < a.max_drop_frac, "too many rows exceed max_seq_len"

    batches_per_epoch = make_batches(rows, a.token_budget, a.micro_cap, rng, a.label_budget)
    micro_per_step = max(1, round(a.accum_rows / (len(rows) / len(batches_per_epoch))))
    steps_per_epoch = math.ceil(len(batches_per_epoch) / micro_per_step)
    total_steps = int(steps_per_epoch * a.epochs)
    print(f"[plan] micro_batches/epoch={len(batches_per_epoch)} micro_per_step={micro_per_step} "
          f"opt_steps={total_steps} avg_rows/micro={len(rows)/len(batches_per_epoch):.1f}", flush=True)
    if a.dry_run:
        ex = rows[0]
        print("---- example prompt tail ----")
        print(repr(tok.decode(ex[0][-200:])))
        print("---- example completion ----")
        print(repr(tok.decode(ex[1])))
        return

    model = AutoModelForCausalLM.from_pretrained(
        a.init, dtype=torch.bfloat16, attn_implementation=a.attn,
    ).cuda()
    # text-only task: the vision tower and projector never see a gradient
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    fwd, lm_head = get_text_model(model)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable params {sum(p.numel() for p in params)/1e9:.2f}B", flush=True)

    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=a.weight_decay, fused=True)
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup * total_steps), total_steps)

    pad_id = tok.pad_token_id or 0
    t0 = time.time()
    step, micro_i, ep = 0, 0, 0
    batches = batches_per_epoch
    run_loss, run_tok = 0.0, 0
    done = False
    while not done:
        for bi in range(0, len(batches), micro_per_step):
            group = batches[bi: bi + micro_per_step]
            # number of supervised tokens in this optimizer step (for correct mean loss)
            denom = sum(sum(len(rows[i][1]) for i in b) for b in group)
            for b in group:
                ids, att, lab = collate(rows, b, pad_id, "cuda")
                h = fwd(ids, att)
                # shift: predict token t+1 from position t
                hs = h[:, :-1, :]
                ys = lab[:, 1:]
                m = ys != -100
                sel = hs[m]                      # [n_label, hidden]
                tgt = ys[m]
                logits = lm_head(sel).float()
                loss = F.cross_entropy(logits, tgt, reduction="sum") / denom
                loss.backward()
                run_loss += loss.item() * denom
                run_tok += int(m.sum())
                del h, hs, sel, logits, loss
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                el = time.time() - t0
                print(f"[step {step}/{total_steps}] loss={run_loss/max(1,run_tok):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} gnorm={gn:.2f} "
                      f"elapsed={el/60:.1f}m eta={(total_steps-step)*el/step/60:.1f}m "
                      f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
                run_loss, run_tok = 0.0, 0
            if a.save_every and step % a.save_every == 0 and step < total_steps:
                save(model, tok, f"{a.out}/checkpoint-{step}")
            if step >= total_steps:
                done = True
                break
        ep += 1
        if not done:
            batches = make_batches(rows, a.token_budget, a.micro_cap, rng, a.label_budget)
    save(model, tok, f"{a.out}/final")
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


def save(model, tok, path):
    os.makedirs(path, exist_ok=True)
    model.config.use_cache = True
    # exp-03: vLLM takes its default sampling params from generation_config.json and
    # evaluate.py overrides nothing, so this file IS the decode policy at grading time.
    # Greedy measured 0.7533 vs 0.5933 for the inherited temperature-1.0 nucleus config.
    # GenerationConfig.save_pretrained REFUSES do_sample=False together with
    # temperature=0.0, and that pair is exactly what we write into the json below - so a
    # checkpoint saved by this script cannot be re-saved by it until the in-memory config
    # is put back into a valid state. (This killed exp-05's first launch at step 300.)
    model.generation_config.do_sample = True
    model.generation_config.temperature = None
    model.generation_config.top_k = None
    model.generation_config.top_p = None
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    # keep the processor/vision side-cars so the dir loads exactly like the base repo
    import shutil
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(path, f)):
            shutil.copy(src, path)
    # Written explicitly rather than through GenerationConfig.save_pretrained, which drops
    # fields equal to its own defaults. vLLM only honours the keys it finds here
    # (temperature/top_p/top_k/...); it does NOT read do_sample, so temperature must be present.
    gc_path = os.path.join(path, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    model.config.use_cache = False
    print(f"[save] {path} generation_config={gc}", flush=True)


if __name__ == "__main__":
    main()
