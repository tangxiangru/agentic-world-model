#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style CoT.

Design notes (each one is a pitfall from skills/exp_protocol/pitfalls.yaml):
  * template_unreachable: the chat template is read from templates/gemma3.jinja -- the very
    file evaluate.py hands to `vllm serve` -- and its md5 is asserted, so training and
    grading render byte-identical strings.
  * eos_mismatch: every target ends with <end_of_turn> (id 106), which is in the base
    generation_config's eos_token_id, so vLLM stops exactly where training says to stop.
  * seq_len_truncation: rows longer than --max-seq-len are dropped, not truncated, and the
    dropped count is printed.
  * logits memory: gemma-3's vocab is 262208, so materialising logits for every position
    would need tens of GB. We run the text tower, then apply lm_head only at the positions
    that carry loss.
Precision: fp32 master weights + bf16 autocast + 8-bit Adam. Pure-bf16 weights would swallow
lr-scale updates (bf16 has ~3 decimal digits) -- fp32 masters avoid that for ~40 GB.
"""
import argparse, hashlib, json, math, os, random, sys, time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

TASK = "/home/ben/task"
TEMPLATE = f"{TASK}/templates/gemma3.jinja"
TEMPLATE_MD5 = "acabb12fa812ef3ab334ea6b817562f3"
EOT = "<end_of_turn>"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_rows(path, tok, max_seq_len, limit=None):
    rows, dropped = [], 0
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            msgs = []
            if r.get("system"):
                msgs.append({"role": "system", "content": r["system"]})
            msgs.append({"role": "user", "content": r["user"]})
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            completion = r["target"]
            assert completion.endswith(EOT), "target must already carry the stop token"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(completion, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) > max_seq_len:
                dropped += 1
                continue
            assert c_ids[-1] == 106, c_ids[-5:]
            rows.append((p_ids, c_ids))
    return rows, dropped


def make_microbatches(rows, token_cap, seed):
    """Length-bucketed micro-batches: pad waste stays small and B*T <= token_cap."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(rows[i][0]) + len(rows[i][1])
        m = max(cur_max, L)
        if cur and m * (len(cur) + 1) > token_cap:
            batches.append(cur)
            cur, cur_max = [i], L
        else:
            cur.append(i)
            cur_max = m
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


def collate(rows, idxs, pad_id, device):
    seqs = [rows[i][0] + rows[i][1] for i in idxs]
    T = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    labels = torch.full((len(seqs), T), -100, dtype=torch.long)
    for b, i in enumerate(idxs):
        p, c = rows[i]
        s = p + c
        input_ids[b, : len(s)] = torch.tensor(s)
        attn[b, : len(s)] = 1
        labels[b, len(p) : len(s)] = torch.tensor(c)
    return (input_ids.to(device, non_blocking=True), attn.to(device, non_blocking=True),
            labels.to(device, non_blocking=True))


def save_bf16(model, tok, d):
    """Save bf16 weights so vLLM loads the same dtype the base checkpoint shipped."""
    log(f"saving {d}")
    sd = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.config.torch_dtype = "bfloat16"
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = "bfloat16"
    model.save_pretrained(d, state_dict=sd, safe_serialization=True)
    tok.save_pretrained(d)
    del sd
    # vLLM builds a Gemma3Processor for Gemma3ForConditionalGeneration and refuses to start
    # without these two; save_pretrained does not write them (pitfall: final_model_not_loadable)
    import shutil
    snap = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
    for f in ("preprocessor_config.json", "processor_config.json"):
        shutil.copyfile(os.path.join(snap, f), os.path.join(d, f))
    # greedy decoding: vLLM takes its default sampling params from this file (adopted in exp-03,
    # worth +4.7 pts). Without it the base's do_sample/top_k/top_p defaults make the grader sample.
    json.dump({"bos_token_id": 2, "cache_implementation": "hybrid", "eos_token_id": [1, 106],
               "pad_token_id": 0, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
              open(os.path.join(d, "generation_config.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{TASK}/data/sft_v1.jsonl")
    ap.add_argument("--parent", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    ap.add_argument("--out", default=f"{TASK}/ckpts/exp-02")
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--token-cap", type=int, default=12288)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-every-frac", type=float, default=None,
                    help="also save an intermediate checkpoint at this fraction of training")
    args = ap.parse_args()

    assert hashlib.md5(open(TEMPLATE, "rb").read()).hexdigest() == TEMPLATE_MD5, \
        "templates/gemma3.jinja changed -- training would render differently from the grader"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.parent)
    tok.chat_template = open(TEMPLATE).read()

    log("tokenizing...")
    rows, dropped = build_rows(args.data, tok, args.max_seq_len, args.limit)
    lens = sorted(len(p) + len(c) for p, c in rows)
    n_tok = sum(lens)
    n_loss = sum(len(c) for _, c in rows)
    log(f"rows={len(rows)} dropped_too_long={dropped} tokens={n_tok/1e6:.1f}M "
        f"loss_tokens={n_loss/1e6:.1f}M p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")

    log("loading model (fp32 masters)...")
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.float32, attn_implementation="sdpa")
    model.model.vision_tower.requires_grad_(False)
    model.model.multi_modal_projector.requires_grad_(False)
    model.cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    text = model.model.language_model
    lm_head = model.lm_head

    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params {sum(p.numel() for p in params)/1e9:.2f}B")

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay, eps=1e-8)

    batches = make_microbatches(rows, args.token_cap, args.seed)
    n_epochs_int = max(1, math.ceil(args.epochs))
    plan = []
    for e in range(n_epochs_int):
        b = make_microbatches(rows, args.token_cap, args.seed + e)
        plan.extend(b)
    plan = plan[: int(len(batches) * args.epochs)]
    total_steps = len(plan) // args.grad_accum
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * args.warmup), total_steps)
    log(f"micro-batches={len(plan)} grad_accum={args.grad_accum} optimizer_steps={total_steps}")

    save_at = int(total_steps * args.save_every_frac) if args.save_every_frac else -1

    t0 = time.time()
    run_loss, run_n = 0.0, 0
    step = 0
    opt.zero_grad(set_to_none=True)
    for mb_i, idxs in enumerate(plan):
        input_ids, attn, labels = collate(rows, idxs, tok.pad_token_id, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = text(input_ids=input_ids, attention_mask=attn).last_hidden_state
            sel = labels[:, 1:] != -100
            h = hs[:, :-1, :][sel]                     # (N, H) only positions that carry loss
            tgt = labels[:, 1:][sel]                   # (N,)
            loss_sum = h.new_zeros((), dtype=torch.float32)
            for k in range(0, h.shape[0], 4096):       # chunk the 262k-way softmax
                lg = lm_head(h[k : k + 4096]).float()
                loss_sum = loss_sum + F.cross_entropy(lg, tgt[k : k + 4096], reduction="sum")
            loss = loss_sum / tgt.numel()
        (loss / args.grad_accum).backward()
        run_loss += loss.item() * tgt.numel()
        run_n += tgt.numel()

        if (mb_i + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                log(f"step {step}/{total_steps} loss {run_loss/max(run_n,1):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {el/60:.1f}min "
                    f"eta {(el/step*(total_steps-step))/60:.1f}min "
                    f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G")
                run_loss, run_n = 0.0, 0
            if step == save_at:
                save_bf16(model, tok, f"{args.out}/mid")

    save_bf16(model, tok, f"{args.out}/final")
    log("done", (time.time() - t0) / 3600, "h")


if __name__ == "__main__":
    main()
