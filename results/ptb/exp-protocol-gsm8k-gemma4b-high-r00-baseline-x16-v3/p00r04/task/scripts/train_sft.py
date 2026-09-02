"""Completion-only SFT of google/gemma-3-4b-pt on grader-format GSM8K targets.

Prompt strings come from scripts/fmt.py, which renders templates/gemma3.jinja --
the same file evaluate.py hands to vLLM -- so train and grade cannot drift.
Loss is masked over the prompt; only the assistant turn (reasoning + the single
ANSWER line + <end_of_turn>) carries gradient.

Two things make this fit an 80 GB H100 at 4 B params with a 262 k vocab:
  * the LM head is evaluated only at the positions that actually carry a label,
    in gradient-checkpointed chunks, so the fp32 logit tensor never exceeds
    chunk x 262144 floats (the naive path tried to allocate 20 GB and OOMed);
  * micro-batches are formed to a token budget rather than a fixed row count,
    because row lengths span 100 - 2800 tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import fmt as F  # noqa: E402

IGNORE = -100


class Rows(torch.utils.data.Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i]}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = m - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [IGNORE] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Length-bucketed micro-batches capped at `max_tokens` padded tokens each."""

    def __init__(self, lengths, max_tokens, seed=0, mega=512):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.seed = seed
        self.mega = mega
        self.epoch = 0
        self._batches = self._build(seed)

    def _build(self, seed):
        rng = random.Random(seed)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.mega):
            chunk = sorted(idx[s: s + self.mega], key=lambda i: self.lengths[i])
            cur, curmax = [], 0
            for i in chunk:
                nm = max(curmax, self.lengths[i])
                if cur and nm * (len(cur) + 1) > self.max_tokens:
                    batches.append(cur)
                    cur, curmax = [i], self.lengths[i]
                else:
                    cur, curmax = cur + [i], nm
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, e):
        self.epoch = e
        self._batches = self._build(self.seed + e)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def encode(tok, rows, max_seq_len, report):
    ids_all, lab_all = [], []
    n_trunc = 0
    lens = []
    for r in rows:
        p = F.render_prompt(tok, r["question"], fewshot=bool(r.get("fewshot")))
        p_ids = tok(p, add_special_tokens=False).input_ids
        t_ids = tok(r["target"], add_special_tokens=False).input_ids
        ids = p_ids + t_ids
        lens.append(len(ids))
        if len(ids) > max_seq_len:
            n_trunc += 1
            continue
        ids_all.append(ids)
        lab_all.append([IGNORE] * len(p_ids) + t_ids)
    lens = np.array(lens)
    kept = np.array([len(x) for x in ids_all])
    report.update(
        n_rows=len(rows), n_kept=len(ids_all), n_dropped=n_trunc,
        drop_frac=float(n_trunc) / max(1, len(rows)),
        len_p50=int(np.percentile(lens, 50)), len_p99=int(np.percentile(lens, 99)),
        len_max=int(lens.max()), total_tokens=int(kept.sum()),
        loss_tokens=int(sum(sum(1 for x in l if x != IGNORE) for l in lab_all)),
    )
    return ids_all, lab_all


def make_trainer_cls(chunk_tokens: int):
    from transformers import Trainer

    class ChunkedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            core = model.model if hasattr(model, "lm_head") else model.module.model
            head = model.lm_head if hasattr(model, "lm_head") else model.module.lm_head
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = core(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )[0]
            sh = h[:, :-1, :]
            sl = labels[:, 1:]
            mask = sl != IGNORE
            sel_h = sh[mask]
            sel_l = sl[mask]
            n = sel_l.numel()

            def piece(hh, ll):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = head(hh)
                return Fn.cross_entropy(lg.float(), ll, reduction="sum")

            total = None
            for i in range(0, n, chunk_tokens):
                part = torch.utils.checkpoint.checkpoint(
                    piece, sel_h[i: i + chunk_tokens], sel_l[i: i + chunk_tokens],
                    use_reentrant=False,
                )
                total = part if total is None else total + part
            loss = total / max(1, n)
            return (loss, None) if return_outputs else loss

        def get_train_dataloader(self):
            from torch.utils.data import DataLoader

            return DataLoader(
                self.train_dataset,
                batch_sampler=self._batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )

    return ChunkedLossTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=F.SNAPSHOT)
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--max-tokens-per-batch", type=int, default=8192)
    ap.add_argument("--chunk-tokens", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-epochs", action="store_true")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(args.model_path)
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]

    report = {}
    ids, labs = encode(tok, rows, args.max_seq_len, report)
    report["template_sha"] = F.template_sha()
    report["stop_token_id"] = tok.convert_tokens_to_ids(F.STOP_TOKEN)
    report["all_targets_end_with_stop"] = all(l[-1] == report["stop_token_id"] for l in labs)
    report["min_loss_tokens"] = int(min(sum(1 for x in l if x != IGNORE) for l in labs))
    print(json.dumps(report, indent=1), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "data_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # The vision tower is dead weight for text-only GSM8K: freeze it so no
    # optimizer state is allocated for it, but keep it in the checkpoint so the
    # saved config stays identical to the base one and vLLM loads it unchanged.
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_train/1e6:.0f}M", flush=True)

    batch_sampler = TokenBudgetBatches(
        [len(x) for x in ids], args.max_tokens_per_batch, seed=args.seed
    )
    steps_per_epoch = math.ceil(len(batch_sampler) / args.grad_accum)
    print(f"micro-batches/epoch {len(batch_sampler)}  optim steps/epoch {steps_per_epoch}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,  # unused: a batch_sampler supplies the batches
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=False,  # autocast is applied explicitly inside compute_loss
        logging_steps=10,
        save_strategy="epoch" if args.save_epochs else ("steps" if args.save_steps else "no"),
        save_steps=args.save_steps or 10 ** 9,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        max_grad_norm=1.0,
        accelerator_config={"dispatch_batches": False, "split_batches": False},
    )
    Cls = make_trainer_cls(args.chunk_tokens)
    trainer = Cls(
        model=model,
        args=targs,
        train_dataset=Rows(ids, labs),
        data_collator=Collator(tok.pad_token_id),
    )
    trainer._batch_sampler = batch_sampler
    res = trainer.train()
    print(res, flush=True)

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    # save in bf16: the base config declares bfloat16 and the grader gives vLLM
    # only 30% of the GPU, where an fp32 copy of a 4 B model would not fit.
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = "bfloat16"
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(args.model_path).save_pretrained(final)
    except Exception as e:  # pragma: no cover
        print("processor save skipped:", e)
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump({"metrics": res.metrics, "data_report": report, "args": vars(args)}, f, indent=1)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
