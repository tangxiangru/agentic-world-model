#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt, rendered with the grader's own chat template.

Every row is rendered with templates/gemma3.jinja (byte-for-byte the file evaluate.py
hands to vLLM) so training and grading see the same string, and every target ends with
"\\n\\nANSWER: <n>" followed by <end_of_turn> (token 106, one of the two eos ids in the
base generation_config, so vLLM stops there).

Mixed precision: fp32 master weights + bf16 autocast + 8-bit AdamW.
Loss: liger fused-linear-cross-entropy (the 262k-token vocab makes materialised logits
the memory bottleneck otherwise).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BASE = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"

# inspect_evals/gsm8k MATH_PROMPT_TEMPLATE, verbatim (already .strip()ed as the task does)
MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

STOP = "<end_of_turn>"


def build_fewshot_pool():
    """gsm8k TRAIN split, formatted exactly like inspect's sample_to_fewshot."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for r in ds:
        parts = r["answer"].split("####")
        pool.append(
            f"{r['question']}\n\nReasoning:\n" + "####".join(parts[:-1]).strip() + f"\n\nANSWER: {parts[-1].strip()}"
        )
    return pool


class FLCETrainer(Trainer):
    _prof = None

    """Trainer whose loss is computed without materialising the [B,T,262208] logits."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )

        self.flce = LigerFusedLinearCrossEntropyLoss(reduction="mean", ignore_index=-100)
        # we return a plain mean-over-tokens loss, so let Trainer divide by accum steps
        self.model_accepts_loss_kwargs = False

    def training_step(self, *a, **kw):
        import time as _t
        if self._prof is None:
            self._prof = {"n": 0, "step": 0.0, "gap": 0.0, "tok": 0, "last": _t.time()}
        p = self._prof
        t0 = _t.time()
        p["gap"] += t0 - p["last"]
        r = super().training_step(*a, **kw)
        torch.cuda.synchronize()
        t1 = _t.time()
        p["step"] += t1 - t0
        p["last"] = t1
        p["n"] += 1
        p["tok"] += int(a[1]["input_ids"].numel())
        if p["n"] % 150 == 0:
            print(f"PROF n={p['n']} micro_step={p['step']/p['n']*1000:.0f}ms gap={p['gap']/p['n']*1000:.0f}ms tok/micro={p['tok']/p['n']:.0f} tok/s={p['tok']/(p['step']+p['gap']):.0f}", flush=True)
        return r

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        m = model.module if hasattr(model, "module") else model
        # accelerate attaches autocast to model.forward, and transformers'
        # autocast_smart_context_manager is a nullcontext on cuda, so calling the inner
        # m.model(...) runs the whole fwd/bwd in fp32 (measured 1538 tok/s vs 7700).
        # Enter autocast explicitly.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )
            h = out[0][:, :-1, :]
            y = labels[:, 1:]
            h = h.reshape(-1, h.size(-1))
            y = y.reshape(-1)
            keep = y != -100
            loss = self.flce(m.lm_head.weight, h[keep], y[keep])
        return (loss, out) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--fewshot-frac", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tmpl = open(TEMPLATE_PATH, "rb").read()
    print("template sha256:", hashlib.sha256(tmpl).hexdigest())

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = tmpl.decode()

    rng = random.Random(args.seed)
    fewshot_pool = build_fewshot_pool() if args.fewshot_frac > 0 else []

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} raw rows")

    def render(row, k_shot: int):
        msgs = []
        if k_shot:
            msgs.append({"role": "system", "content": "\n\n".join(rng.sample(fewshot_pool, k_shot))})
        msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=row["question"])})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt, row["target"].strip() + STOP

    p, t = render(rows[0], 0)
    print("=" * 30, "RENDERED (0-shot) prompt:")
    print(repr(p))
    print("-" * 10, "target:")
    print(repr(t))
    p4, _ = render(rows[0], 4)
    print("=" * 30, "RENDERED (4-shot) head/tail:")
    print(repr(p4[:300]), "...", repr(p4[-260:]))

    n_fs = int(len(rows) * args.fewshot_frac)
    ks = [rng.choice([2, 4, 8]) for _ in range(n_fs)] + [0] * (len(rows) - n_fs)
    rng.shuffle(ks)

    feats = {"input_ids": [], "labels": [], "length": []}
    n_drop = 0
    lens, tgt_lens = [], []
    BATCH = 2000
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        kk = ks[start : start + BATCH]
        prompts, fulls = [], []
        for r, k in zip(chunk, kk):
            pp, tt = render(r, k)
            prompts.append(pp)
            fulls.append(pp + tt)
        pid = tok(prompts, add_special_tokens=False)["input_ids"]
        fid = tok(fulls, add_special_tokens=False)["input_ids"]
        for a, b in zip(pid, fid):
            if len(b) > args.max_len or b[: len(a)] != a or len(b) == len(a):
                n_drop += 1
                continue
            feats["input_ids"].append(b)
            feats["labels"].append([-100] * len(a) + b[len(a) :])
            feats["length"].append(len(b))
            lens.append(len(b))
            tgt_lens.append(len(b) - len(a))
        if start % 40000 == 0:
            print(f"tokenized {start + len(chunk)}", flush=True)

    slens = sorted(lens)
    n = len(slens)
    stats = {
        "n_rows": n,
        "dropped": n_drop,
        "p50": slens[n // 2],
        "p99": slens[int(n * 0.99)],
        "max": slens[-1],
        "total_tokens": sum(lens),
        "total_target_tokens": sum(tgt_lens),
        "max_len": args.max_len,
    }
    print("DATASTATS", json.dumps(stats))
    os.makedirs(args.out, exist_ok=True)
    json.dump(stats, open(os.path.join(args.out, "datastats.json"), "w"), indent=2)

    # eos sanity: every row's last token must be <end_of_turn>
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    bad = sum(1 for x in feats["input_ids"] if x[-1] != eot)
    print(f"rows not ending in <end_of_turn>({eot}): {bad}")
    assert bad == 0

    if args.dry_run:
        return

    ds = Dataset.from_dict(feats)
    pad_id = tok.pad_token_id

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        m = (m + 7) // 8 * 8
        ii, ll, am = [], [], []
        for b in batch:
            k = m - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad_id] * k)
            ll.append(b["labels"] + [-100] * k)
            am.append([1] * len(b["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ii, dtype=torch.long),
            "labels": torch.tensor(ll, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
        }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.parent, dtype=torch.float32, attn_implementation=args.attn
        )
    except Exception as e:  # pragma: no cover
        print("attn impl failed, falling back to sdpa:", e)
        model = AutoModelForCausalLM.from_pretrained(args.parent, dtype=torch.float32, attn_implementation="sdpa")
    print("model class:", type(model).__name__)
    model.config.use_cache = False
    for name, p_ in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p_.requires_grad_(False)
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"trainable {n_train/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 1000000,
        save_total_limit=2,
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = FLCETrainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    model.to(torch.bfloat16)  # vLLM must load bf16, not the fp32 master weights
    trainer.save_model(final)
    tok.save_pretrained(final)
    for fn in ["preprocessor_config.json", "processor_config.json", "tokenizer.model"]:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "transformers_version": "4.50.0.dev0",
    }
    json.dump(gc, open(os.path.join(final, "generation_config.json"), "w"), indent=2)
    print("saved", final)


if __name__ == "__main__":
    main()
