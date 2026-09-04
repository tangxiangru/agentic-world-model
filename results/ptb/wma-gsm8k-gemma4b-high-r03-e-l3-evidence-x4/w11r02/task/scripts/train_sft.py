"""Completion-only SFT for google/gemma-3-4b-pt on the graded GSM8K prompt.

Rows are rendered with scripts/fmt.py, i.e. with templates/gemma3.jinja - the
same file evaluate.py hands to vLLM - so training and grading cannot disagree
about the string (pitfalls.yaml: template_unreachable). Loss is taken only on
the assistant turn, which always ends with <end_of_turn> (eos_mismatch).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


class SFTRows(Dataset):
    def __init__(self, path, tokenizer, max_seq_len, fewshot_frac, seed,
                 fewshot_choices=(10,), limit=None):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        rng = random.Random(seed)
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]
        self.items = []
        self.n_trunc = 0
        lens = []
        for r in rows:
            n_shot = rng.choice(fewshot_choices) if rng.random() < fewshot_frac else 0
            prompt = fmt.render_prompt(r["question"], n_shot, tokenizer)
            target = r["completion"]
            assert target.endswith(fmt.END_OF_TURN), "row is not terminated by the stop token"
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            t_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            lens.append(len(p_ids) + len(t_ids))
            if len(p_ids) + len(t_ids) > max_seq_len:
                self.n_trunc += 1
                continue  # never truncate a target; drop the row instead
            self.items.append((p_ids, t_ids))
        lens.sort()
        self.stats = {
            "rows_in": len(rows),
            "rows_kept": len(self.items),
            "rows_dropped_too_long": self.n_trunc,
            "len_p50": lens[len(lens) // 2],
            "len_p99": lens[int(len(lens) * 0.99)],
            "len_max": lens[-1],
            "tokens_total": sum(len(a) + len(b) for a, b in self.items),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, t = self.items[i]
        return {"input_ids": p + t, "labels": [-100] * len(p) + list(t)}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def build_batches(lengths, token_budget, max_rows):
    """Length-grouped micro-batches under a *token* budget, not a row budget.

    The logits tensor is [tokens, 262208] and cross-entropy materialises it in
    fp32 twice, so peak memory tracks total tokens in the micro-batch, not rows.
    A fixed row batch OOMs the moment length-grouping puts eight 2.4k-token
    few-shot rows together (that is exactly how the exp-02 smoke run died).
    Batches are built once and only their order is shuffled, so len() is exact
    and the LR schedule is right.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > token_budget or len(cur) + 1 > max_rows):
            batches.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur.append(i)
            cur_max = m
    if cur:
        batches.append(cur)
    return batches


def _make_trainer_cls():
    from transformers import Trainer

    class BudgetTrainer(Trainer):
        batches: list = []
        batch_seed: int = 0

        def get_train_dataloader(self):
            from torch.utils.data import DataLoader

            g = torch.Generator()
            g.manual_seed(self.batch_seed)
            order = torch.randperm(len(self.batches), generator=g).tolist()
            sampler = [self.batches[i] for i in order]
            return self.accelerator.prepare(
                DataLoader(
                    self.train_dataset,
                    batch_sampler=sampler,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=True,
                )
            )

    return Trainer, BudgetTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--token-budget", type=int, default=2560)
    ap.add_argument("--max-rows", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=24)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--fewshot-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                              TrainingArguments)

    Trainer, BudgetTrainer = _make_trainer_cls()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.fewshot_frac, args.seed)
    print(json.dumps(ds.stats, indent=2), flush=True)

    # dry run: prove the row the trainer will see is the string the grader sends
    p, t = ds.items[0]
    print("---- example prompt tail ----", flush=True)
    print(repr(tok.decode(p[-160:])), flush=True)
    print("---- example target ----", flush=True)
    print(repr(tok.decode(t)), flush=True)
    assert tok.decode(t).endswith(fmt.END_OF_TURN), "target does not end in the stop token"
    if args.dry_run:
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # text-only task: freeze the vision tower and the projector so AdamW never
    # allocates state for ~430M parameters that receive no gradient
    n_frozen = 0
    for name, p_ in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith(
            "model.multi_modal_projector"
        ):
            p_.requires_grad_(False)
            n_frozen += p_.numel()
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M, trainable {n_train/1e6:.0f}M", flush=True)

    batches = build_batches(
        [len(a) + len(b) for a, b in ds.items], args.token_budget, args.max_rows
    )
    steps_per_epoch = math.ceil(len(batches) / args.grad_accum)
    print(
        f"microbatches/epoch={len(batches)} rows/mb p50="
        f"{sorted(len(b) for b in batches)[len(batches)//2]} "
        f"steps/epoch={steps_per_epoch}",
        flush=True,
    )
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=4,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
    )

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.batches = batches
    trainer.batch_seed = args.seed
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # gemma-3 is a multimodal checkpoint: vLLM resolves an AutoProcessor for it,
    # so the processor/preprocessor files have to travel with the weights or the
    # grader cannot load final_model (pitfalls.yaml: final_model_not_loadable)
    import shutil

    for extra in ("preprocessor_config.json", "processor_config.json",
                  "added_tokens.json", "tokenizer.model"):
        src = os.path.join(args.model, extra)
        dst = os.path.join(final, extra)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    # the grader decodes with whatever this file says; make it greedy and keep
    # both stop ids so vLLM stops on <end_of_turn>
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(
            {
                "bos_token_id": 2,
                "eos_token_id": [1, 106],
                "pad_token_id": 0,
                "cache_implementation": "hybrid",
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "transformers_version": "4.57.3",
            },
            f,
            indent=2,
        )
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
