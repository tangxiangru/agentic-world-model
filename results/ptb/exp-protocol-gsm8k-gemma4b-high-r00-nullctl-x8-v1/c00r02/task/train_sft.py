#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt for GSM8K-style math reasoning."""
import argparse, json, math, os, random, time, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def build_prompt(question: str, system: str | None) -> str:
    """Replicate templates/gemma3.jinja exactly (bos handled by tokenizer)."""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    if system:
        user = system.strip() + "\n\n" + user
    return f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"


def load_fewshot_pool():
    from datasets import load_dataset
    import re
    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for r in ds:
        body, final = r["answer"].split("####")
        body = body.strip()
        pool.append(f"{r['question']}\n\nReasoning:\n{body}\n\nANSWER: {final.strip()}")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/sft_data.jsonl")
    ap.add_argument("--out", default="work/sft_v1")
    ap.add_argument("--init", default=SNAPSHOT)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--accum-tokens", type=int, default=131072)
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    EOT = "<end_of_turn>"

    # ---------------- data ----------------
    recs = [json.loads(l) for l in open(args.data)]
    random.shuffle(recs)
    if args.max_examples:
        recs = recs[: args.max_examples]
    fewshot_pool = load_fewshot_pool() if args.fewshot_frac > 0 else []

    print(f"tokenizing {len(recs)} examples...", flush=True)
    examples = []
    t0 = time.time()
    prompts, completions = [], []
    for r in recs:
        system = None
        if fewshot_pool and random.random() < args.fewshot_frac:
            k = random.randint(1, 10)
            shots = random.sample(fewshot_pool, k)
            system = "\n\n".join(shots)
        prompts.append(build_prompt(r["question"], system))
        completions.append(r["solution"].strip() + f"\n\nANSWER: {r['answer']}" + EOT)

    B = 2000
    for i in range(0, len(prompts), B):
        pe = tok(prompts[i:i + B], add_special_tokens=True)["input_ids"]
        ce = tok(completions[i:i + B], add_special_tokens=False)["input_ids"]
        for p, c in zip(pe, ce):
            if len(p) + len(c) > args.max_len:
                continue
            examples.append((p + c, len(p)))
    print(f"tokenized in {time.time()-t0:.0f}s; kept {len(examples)}", flush=True)
    total_tokens = sum(len(e[0]) for e in examples)
    print(f"total tokens {total_tokens/1e6:.1f}M", flush=True)

    # ---------------- batching (length-bucketed) ----------------
    def make_batches(exs, seed):
        rng = random.Random(seed)
        order = list(range(len(exs)))
        rng.shuffle(order)
        # megabatches of 2048 sorted by length -> low padding, still stochastic
        batches = []
        MB = 2048
        for s in range(0, len(order), MB):
            chunk = sorted(order[s:s + MB], key=lambda i: len(exs[i][0]))
            cur, curmax = [], 0
            for i in chunk:
                L = len(exs[i][0])
                nmax = max(curmax, L)
                if cur and nmax * (len(cur) + 1) > args.micro_tokens:
                    batches.append(cur)
                    cur, curmax = [i], L
                else:
                    cur.append(i)
                    curmax = nmax
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    # ---------------- model ----------------
    if not args.no_liger:
        sys.path.insert(0, "/home/ben/task/pylibs")
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger applied", flush=True)

    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation=args.attn,
    ).cuda()
    model.config.use_cache = False
    # freeze vision tower / projector (text-only training)
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") \
           or name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M, trainable {trainable/1e6:.0f}M", flush=True)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()

    import bitsandbytes as bnb
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim == 1 or "norm" in n) else decay).append(p)
    optim = bnb.optim.AdamW8bit(
        [{"params": decay, "weight_decay": 0.0}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    n_epochs_int = math.ceil(args.epochs)
    all_batches = []
    for ep in range(n_epochs_int):
        all_batches.extend(make_batches(examples, args.seed + ep))
    all_batches = all_batches[: int(len(all_batches) * args.epochs / n_epochs_int)]

    # group micro-batches into optimizer steps by token budget
    steps = []
    cur, curtok = [], 0
    for b in all_batches:
        ntok = sum(len(examples[i][0]) for i in b)
        cur.append(b)
        curtok += ntok
        if curtok >= args.accum_tokens:
            steps.append(cur)
            cur, curtok = [], 0
    if cur:
        steps.append(cur)
    print(f"{len(all_batches)} micro-batches, {len(steps)} optimizer steps", flush=True)

    sched = get_cosine_schedule_with_warmup(optim, args.warmup, len(steps))

    def collate(idxs):
        seqs = [examples[i] for i in idxs]
        L = max(len(s[0]) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.long)
        lab = torch.full((len(seqs), L), -100, dtype=torch.long)
        for r, (s, plen) in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
            lab[r, plen:len(s)] = torch.tensor(s[plen:])
        return ids, att, lab

    t0 = time.time()
    seen_tokens = 0
    ema = None
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train_log.txt"), "a")
    for step, group in enumerate(steps):
        # number of supervised tokens in this optimizer step (for correct normalization)
        n_target = sum(
            (len(examples[i][0]) - examples[i][1]) for b in group for i in b
        )
        step_loss = 0.0
        for b in group:
            try:
                ids, att, lab = collate(b)
                ids, att, lab = ids.cuda(), att.cuda(), lab.cuda()
                out = model(input_ids=ids, attention_mask=att, labels=lab)
                # HF returns mean loss over non-masked tokens in this micro-batch
                ntok_mb = (lab != -100).sum()
                loss = out.loss * ntok_mb / n_target
                loss.backward()
                step_loss += loss.item()
                seen_tokens += int(att.sum())
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on micro-batch of {len(b)} seqs, skipping", flush=True)
                del ids, att, lab
                try:
                    del out, loss
                except Exception:
                    pass
                torch.cuda.empty_cache()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        ema = step_loss if ema is None else 0.95 * ema + 0.05 * step_loss
        if step % 5 == 0 or step == len(steps) - 1:
            el = time.time() - t0
            msg = (f"step {step+1}/{len(steps)} loss {step_loss:.4f} ema {ema:.4f} "
                   f"gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} "
                   f"tok/s {seen_tokens/el:.0f} elapsed {el/60:.1f}m "
                   f"eta {(len(steps)-step-1)*el/(step+1)/60:.1f}m")
            print(msg, flush=True)
            logf.write(msg + "\n"); logf.flush()
        if args.save_every and (step + 1) % args.save_every == 0:
            d = os.path.join(args.out, f"ckpt-{step+1}")
            model.save_pretrained(d, safe_serialization=True)
            tok.save_pretrained(d)

    print("saving...", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    # greedy decoding for eval
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump({
            "bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
            "cache_implementation": "hybrid", "do_sample": False, "temperature": 0.0,
            "top_p": 1.0, "top_k": -1,
        }, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
