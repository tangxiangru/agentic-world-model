"""Full-parameter SFT of gemma-3-4b-pt on prompt/completion jsonl.

Design notes (each one is a pitfall being avoided):
  * targets are graded by match(location="end", numeric=True) after the vLLM
    server stops on <end_of_turn>, so every completion ends with the stop token
    and loss is computed on the completion only;
  * prompts are rendered by scripts/fmt.py, which is asserted byte-identical to
    templates/gemma3.jinja (the file the grader passes to vLLM);
  * rows longer than --max-seq-len are dropped, not truncated, and the count is
    printed, so a silently-truncated completion cannot carry zero loss tokens;
  * master weights are fp32 with bf16 autocast: a pure-bf16 update at lr 1e-5 is
    below bf16's relative precision and would be partly rounded away;
  * the lm_head is applied only to label positions - a full [B, T, 262208]
    logits tensor does not fit alongside the optimizer state.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

from transformers import AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_batches(rows, tok, max_seq_len, token_budget, max_bs, seed):
    ex = []
    n_drop = 0
    prompts = [r["prompt"] for r in rows]
    comps = [r["completion"] for r in rows]
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    c_ids = tok(comps, add_special_tokens=False)["input_ids"]
    for p, c in zip(p_ids, c_ids):
        if len(p) + len(c) > max_seq_len:
            n_drop += 1
            continue
        ex.append((p + c, len(p)))
    log(f"tokenized {len(ex)} rows, dropped {n_drop} over max_seq_len={max_seq_len}")

    ex.sort(key=lambda t: len(t[0]))
    batches, cur = [], []
    for e in ex:
        trial = cur + [e]
        if len(trial) > max_bs or len(trial) * len(trial[-1][0]) > token_budget:
            if cur:
                batches.append(cur)
            cur = [e]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches, n_drop


def collate(batch, pad_id, device):
    L = max(len(x[0]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    for i, (ids, np_) in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids)
        attn[i, :n] = 1
        labels[i, np_:n] = torch.tensor(ids[np_:n])
    return (input_ids.to(device, non_blocking=True),
            attn.to(device, non_blocking=True),
            labels.to(device, non_blocking=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--max-bs", type=int, default=48)
    ap.add_argument("--grad-accum", type=int, default=6)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every-frac", type=float, default=0.0,
                    help="also save an intermediate checkpoint at this fraction of training")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--bench-steps", type=int, default=0)
    args = ap.parse_args()

    log("template sha256:", fmt.TEMPLATE_SHA256)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    rows = [json.loads(l) for l in open(args.data)]
    if args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    log(f"rows: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    batches, _ = build_batches(rows, tok, args.max_seq_len, args.token_budget,
                               args.max_bs, args.seed)
    n_epoch_batches = len(batches)
    total_micro = int(n_epoch_batches * args.epochs)
    total_steps = max(1, total_micro // args.grad_accum)
    log(f"micro-batches/epoch {n_epoch_batches}, total micro {total_micro}, "
        f"optimizer steps {total_steps}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="flash_attention_2"
    )
    lm = model.model.language_model
    head = model.lm_head
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or n.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # from_pretrained returns the model in eval mode, and Gemma3TextModel only
    # takes the checkpointing branch when self.training is True -> without this
    # the run OOMs at full activation memory.
    model.train()
    assert lm.gradient_checkpointing and lm.training, "gradient checkpointing inactive"
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params: {sum(p.numel() for p in trainable)/1e9:.3f} B")

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay, eps=1e-8)

    warm = max(1, int(args.warmup * total_steps))

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        prog = min(1.0, prog)
        c = 0.5 * (1 + math.cos(math.pi * prog))
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * c)

    def order():
        e = 0
        while True:
            idx = list(range(n_epoch_batches))
            random.Random(args.seed + e).shuffle(idx)
            for i in idx:
                yield batches[i]
            e += 1

    gen = order()
    micro_list = [next(gen) for _ in range(total_micro)]

    t0 = time.time()
    step = 0
    tokens_seen = 0
    loss_acc, ntok_acc = 0.0, 0
    running = []
    saved_mid = False

    for mi in range(0, total_micro - args.grad_accum + 1, args.grad_accum):
        window = micro_list[mi: mi + args.grad_accum]
        denom = sum(sum(1 for j in range(b[1], len(b[0]))) for mb in window for b in mb)
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        for mb in window:
            input_ids, attn, labels = collate(mb, pad_id, device)
            tokens_seen += int(attn.sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = lm(input_ids=input_ids, attention_mask=attn)
                h = out.last_hidden_state
            hs = h[:, :-1, :].reshape(-1, h.size(-1))
            ls = labels[:, 1:].reshape(-1)
            m = ls != -100
            hs = hs[m]
            ls = ls[m]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(hs)
            loss_sum = F.cross_entropy(logits.float(), ls, reduction="sum")
            (loss_sum / denom).backward()
            loss_acc += float(loss_sum.detach())
            ntok_acc += int(m.sum())
            del out, h, hs, logits, loss_sum
        torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        running.append(loss_acc / max(1, ntok_acc))
        loss_acc, ntok_acc = 0.0, 0
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            log(f"step {step}/{total_steps} loss {sum(running[-10:])/len(running[-10:]):.4f} "
                f"lr {lr:.2e} tok/s {tokens_seen/el:.0f} "
                f"eta_h {(total_steps-step)*el/step/3600:.2f} "
                f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G")
        if args.bench_steps and step >= args.bench_steps:
            log("bench done"); return
        if (args.save_every_frac and not saved_mid
                and step >= int(args.save_every_frac * total_steps)):
            saved_mid = True
            save(model, tok, args.model, args.out + f"/mid-step{step}")

    save(model, tok, args.model, args.out + "/final")
    log(f"done in {(time.time()-t0)/3600:.2f} h")


def save(model, tok, src_model, path):
    os.makedirs(path, exist_ok=True)
    log("saving to", path)
    # save_pretrained runs GenerationConfig.validate(strict=True), which rejects
    # the greedy config we write below (do_sample=False with temperature set).
    # Hand the saver a plain valid config and overwrite the file afterwards.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid")
    model_bf16 = model.to(torch.bfloat16)
    model_bf16.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    try:
        proc = AutoProcessor.from_pretrained(src_model)
        proc.save_pretrained(path)
    except Exception as e:
        log("processor save failed (non-fatal):", e)
    # greedy decoding: vLLM reads generation_config.json as the server's default
    # sampling params, and the base config asks for top_k=64/top_p=0.95 sampling
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "cache_implementation": "hybrid",
                   "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
                  f, indent=2)
    model.to(torch.float32)
    log("saved", path)


if __name__ == "__main__":
    main()
