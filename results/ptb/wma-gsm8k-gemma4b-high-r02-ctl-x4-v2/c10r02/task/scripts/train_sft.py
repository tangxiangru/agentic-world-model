"""Full fine-tune of google/gemma-3-4b-pt for GSM8K.

Prompts are rendered with the *grader's* chat template (templates/gemma3.jinja,
hash-checked), targets end with <end_of_turn>, and the loss is taken on the
completion only.  fp32 master weights + bf16 autocast + 8-bit AdamW.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, get_cosine_schedule_with_warmup

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
TEMPLATE_SHA256 = "cc57b8b1a4b1d1c1"  # filled at runtime, printed for the card

PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response should be of the '
    'form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    "{prompt}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without '
    "quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sft_all.jsonl")
    p.add_argument("--parent", default=BASE)
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-seq-len", type=int, default=3072)
    p.add_argument("--tokens-per-micro", type=int, default=8192)
    p.add_argument("--tokens-per-step", type=int, default=131072)
    p.add_argument("--full-fewshot-frac", type=float, default=0.10)
    p.add_argument("--part-fewshot-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every-frac", type=float, default=0.0,
                   help="if >0, also save an intermediate checkpoint at this fraction of training")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--tokenize-only", action="store_true")
    p.add_argument("--liger", type=int, default=1)
    p.add_argument("--max-hours", type=float, default=100.0)
    return p.parse_args()


def build_rows(args, tok, template, fewshot_full, fewshot_parts, rng):
    rows = []
    with open(args.data) as fh:
        data = [json.loads(l) for l in fh]
    if args.n_train > 0:
        data = data[: args.n_train]
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    n_trunc = 0
    for rec in data:
        r = rng.random()
        if r < args.full_fewshot_frac:
            sys_msg = fewshot_full
        elif r < args.full_fewshot_frac + args.part_fewshot_frac:
            sys_msg = rng.choice(fewshot_parts)
        else:
            sys_msg = None
        msgs = []
        if sys_msg is not None:
            msgs.append({"role": "system", "content": sys_msg})
        msgs.append({"role": "user", "content": PROMPT_TEMPLATE.format(prompt=rec["question"])})
        prompt_text = tok.apply_chat_template(msgs, chat_template=template, tokenize=False,
                                              add_generation_prompt=True)
        p_ids = tok(prompt_text, add_special_tokens=False).input_ids
        c_ids = tok(rec["target"].strip(), add_special_tokens=False).input_ids + [eot]
        if len(p_ids) + len(c_ids) > args.max_seq_len:
            n_trunc += 1
            continue
        rows.append((p_ids, c_ids))
    print(f"tokenized {len(rows)} rows, dropped {n_trunc} over max_seq_len={args.max_seq_len}", flush=True)
    return rows


def make_micro_batches(rows, tokens_per_micro, rng):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(rows[i][0]) + len(rows[i][1])
        m = max(cur_max, L)
        if cur and m * (len(cur) + 1) > tokens_per_micro:
            batches.append(cur)
            cur, cur_max = [i], L
        else:
            cur.append(i)
            cur_max = m
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(rows, idxs, device, pad_id):
    seqs = [(rows[i][0] + rows[i][1]) for i in idxs]
    labs = [([-100] * len(rows[i][0])) + rows[i][1] for i in idxs]
    L = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    labels = torch.full((len(seqs), L), -100, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, (s, lb) in enumerate(zip(seqs, labs)):
        input_ids[j, : len(s)] = torch.tensor(s)
        labels[j, : len(lb)] = torch.tensor(lb)
        attn[j, : len(s)] = 1
    return input_ids.to(device), labels.to(device), attn.to(device)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    template = open(TEMPLATE_PATH).read()
    tsha = hashlib.sha256(template.encode()).hexdigest()
    print("chat template sha256:", tsha, flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    fewshot_full = open("data/fewshot_system.txt").read()
    shots = fewshot_full.split("\n\n\n") if "\n\n\n" in fewshot_full else None
    # the eval system message is 10 exemplars joined by "\n\n"; each exemplar itself
    # contains "\n\n", so split on the ANSWER line instead.
    parts, buf = [], []
    for block in fewshot_full.split("\n\n"):
        buf.append(block)
        if block.startswith("ANSWER: "):
            parts.append("\n\n".join(buf))
            buf = []
    assert len(parts) == 10, len(parts)
    fewshot_parts = ["\n\n".join(parts[:k]) for k in (2, 3, 4)]

    rows = build_rows(args, tok, template, fewshot_full, fewshot_parts, rng)
    tot_tokens = sum(len(a) + len(b) for a, b in rows)
    lab_tokens = sum(len(b) for a, b in rows)
    print(f"total tokens/epoch {tot_tokens/1e6:.1f}M, label tokens {lab_tokens/1e6:.1f}M", flush=True)
    if args.tokenize_only:
        return

    print("loading model...", flush=True)
    if args.liger:
        from liger_kernel.transformers import monkey_patch as lmp
        lmp.apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True, cross_entropy=False)
        print("liger fused-linear-CE enabled", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.float32, attn_implementation="sdpa")
    model.model.vision_tower.requires_grad_(False)
    model.model.multi_modal_projector.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.cuda()
    model.train()  # gradient checkpointing is a no-op unless the module is in training mode
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in trainable)/1e9:.2f}B", flush=True)

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay, eps=1e-8)

    micro = make_micro_batches(rows, args.tokens_per_micro, rng)
    # group micro-batches into optimizer steps by token budget
    steps, cur, cur_tok = [], [], 0
    for mb in micro:
        L = max(len(rows[i][0]) + len(rows[i][1]) for i in mb)
        cur.append(mb)
        cur_tok += L * len(mb)
        if cur_tok >= args.tokens_per_step:
            steps.append(cur)
            cur, cur_tok = [], 0
    if cur:
        steps.append(cur)
    n_epoch_steps = len(steps)
    total_steps = int(n_epoch_steps * args.epochs)
    print(f"{len(micro)} micro-batches, {n_epoch_steps} steps/epoch, {total_steps} total steps", flush=True)

    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    pad_id = tok.pad_token_id
    t0 = time.time()
    done = 0
    logf = open(os.path.join(args.out, "train_log.jsonl"), "a")
    saved_mid = False
    stop = False
    for ep in range(math.ceil(args.epochs)):
        if ep > 0:
            micro = make_micro_batches(rows, args.tokens_per_micro, rng)
            steps, cur, cur_tok = [], [], 0
            for mb in micro:
                L = max(len(rows[i][0]) + len(rows[i][1]) for i in mb)
                cur.append(mb)
                cur_tok += L * len(mb)
                if cur_tok >= args.tokens_per_step:
                    steps.append(cur)
                    cur, cur_tok = [], 0
            if cur:
                steps.append(cur)
        for step_mbs in steps:
            if done >= total_steps:
                stop = True
                break
            n_lab = sum(sum(len(rows[i][1]) for i in mb) for mb in step_mbs)
            step_loss = 0.0
            for mb in step_mbs:
                input_ids, labels, attn = collate(rows, mb, "cuda", pad_id)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.liger:
                        out = model(input_ids=input_ids, attention_mask=attn, labels=labels,
                                    num_items_in_batch=n_lab)
                        loss = out.loss  # already sum / n_lab
                    else:
                        out = model(input_ids=input_ids, attention_mask=attn)
                        logits = out.logits[:, :-1, :]
                        tgt = labels[:, 1:]
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                               tgt.reshape(-1), ignore_index=-100,
                                               reduction="sum") / n_lab
                loss.backward()
                step_loss += loss.item() * n_lab
                del out, loss
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            if done % args.log_every == 0 or done == 1:
                el = (time.time() - t0) / 3600
                rec = {"step": done, "total": total_steps, "loss": step_loss / n_lab,
                       "lr": sched.get_last_lr()[0], "grad_norm": float(gn), "hours": round(el, 3),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            if args.save_every_frac > 0 and not saved_mid and done >= int(total_steps * args.save_every_frac):
                saved_mid = True
                save(model, tok, os.path.join(args.out, f"checkpoint-{done}"))
            if (time.time() - t0) / 3600 > args.max_hours:
                print("time budget hit, stopping", flush=True)
                stop = True
                break
        if stop:
            break

    save(model, tok, os.path.join(args.out, "final"))
    print("done in %.2f h" % ((time.time() - t0) / 3600), flush=True)


def save(model, tok, path):
    """Write a bf16 copy without disturbing the live fp32 master weights."""
    print("saving", path, flush=True)
    os.makedirs(path, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    old = model.config.use_cache
    model.config.use_cache = True
    model.save_pretrained(path, safe_serialization=True, state_dict=sd)
    model.config.use_cache = old
    tok.save_pretrained(path)
    del sd
    # Gemma3ForConditionalGeneration is a multimodal architecture: vLLM refuses to
    # start without the image-processor configs, even for a text-only workload.
    import shutil
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(path, fn))
    # the live model is fp32 (master weights); the saved copy is bf16, so the
    # config on disk must say bf16 or vLLM will materialise 17 GB of fp32.
    cfgp = os.path.join(path, "config.json")
    with open(cfgp) as fh:
        cfg = json.load(fh)

    def fix(d):
        for k in ("torch_dtype", "dtype"):
            if k in d:
                d[k] = "bfloat16"
        for v in d.values():
            if isinstance(v, dict):
                fix(v)
    fix(cfg)
    cfg["torch_dtype"] = "bfloat16"
    with open(cfgp, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
