#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt for GSM8K.

Renders every row with the *grader's own* chat template (templates/gemma3.jinja),
so training and grading agree byte-for-byte, and terminates every target with
<end_of_turn> (token 106, which is in the base generation_config's eos list).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoProcessor,
    GenerationConfig,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"


class TokenBudgetBatchSampler(Sampler):
    """Length-bucketed batches with a fixed token budget.

    Gemma-3's 262k vocab makes the fp32 logits upcast the memory bottleneck
    (batch x seq x 262144 x 4 bytes), so batches must be capped by *tokens*,
    not by rows: a fixed row count OOMs as soon as a long-row bucket comes up.
    """

    def __init__(self, lengths: list[int], max_tokens: int, seed: int = 0):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.seed = seed
        self.epoch = 0
        self.batches = self._build(seed)

    def _build(self, seed: int) -> list[list[int]]:
        rng = random.Random(seed)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        # megabatch sort: local length sorting keeps padding low without
        # making the sample order deterministic across epochs
        mega = 4096
        order: list[int] = []
        for i in range(0, len(idx), mega):
            order.extend(sorted(idx[i : i + mega], key=lambda j: self.lengths[j]))
        batches, cur, curmax = [], [], 0
        for j in order:
            m = max(curmax, self.lengths[j])
            if cur and m * (len(cur) + 1) > self.max_tokens:
                batches.append(cur)
                cur, curmax = [j], self.lengths[j]
            else:
                cur.append(j)
                curmax = m
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.batches = self._build(self.seed + epoch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def build_fewshot_prefix(n: int = 10, seed: int = 42) -> str:
    """The exact system message inspect_evals/gsm8k builds (10-shot, seed 42)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed).select(range(n))
    parts = []
    for r in ds:
        body, final = r["answer"].rsplit("####", 1)
        import re

        body = re.sub(r"<<[^>]*>>", "", body).strip()
        parts.append(f"{r['question']}\n\nReasoning:\n{body}\n\nANSWER: {final.strip()}")
    return "\n\n".join(parts)


def render(prompt: str, target: str, system: str | None) -> tuple[str, str]:
    """Return (prompt_string, full_string) exactly as templates/gemma3.jinja would."""
    first = f"{system.strip()}\n\n" if system else ""
    p = f"{BOS}{SOT}user\n{first}{prompt.strip()}{EOT}\n{SOT}model\n"
    # the data file already terminates every target with <end_of_turn>
    t = target.strip()
    if not t.endswith(EOT):
        t += EOT
    return p, p + t


def verify_template(tok, template_path: str) -> None:
    """Fail loudly if our hand-rolled render() disagrees with the grader's jinja."""
    with open(template_path) as f:
        tmpl = f.read()
    msgs = [
        {"role": "system", "content": "SYS TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]
    ref = tok.apply_chat_template(
        msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True
    )
    ours, _ = render("USER TEXT", "x", "SYS TEXT")
    assert ref == ours, f"template mismatch\nref ={ref!r}\nours={ours!r}"
    msgs2 = [{"role": "user", "content": "USER TEXT"}]
    ref2 = tok.apply_chat_template(
        msgs2, chat_template=tmpl, tokenize=False, add_generation_prompt=True
    )
    ours2, _ = render("USER TEXT", "x", None)
    assert ref2 == ours2, f"template mismatch\nref ={ref2!r}\nours={ours2!r}"
    print("template check ok; sha256", hashlib.sha256(tmpl.encode()).hexdigest()[:16])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="templates/gemma3.jinja")
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--fewshot-frac", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    verify_template(tok, args.template)
    eot_id = tok.convert_tokens_to_ids(EOT)
    print("EOT id", eot_id)

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    rng = random.Random(args.seed)
    fewshot = build_fewshot_prefix()
    print("fewshot prefix chars", len(fewshot))

    feats = {"input_ids": [], "labels": [], "length": []}
    n_trunc = 0
    n_fs = 0
    for r in rows:
        use_fs = rng.random() < args.fewshot_frac
        sysmsg = fewshot if use_fs else None
        p, full = render(r["prompt"], r["target"], sysmsg)
        pid = tok(p, add_special_tokens=False)["input_ids"]
        fid = tok(full, add_special_tokens=False)["input_ids"]
        assert fid[: len(pid)] == pid
        assert fid[-1] == eot_id, "target does not end with <end_of_turn>"
        if len(fid) > args.max_seq_len:
            n_trunc += 1
            continue
        lab = [-100] * len(pid) + fid[len(pid) :]
        feats["input_ids"].append(fid)
        feats["labels"].append(lab)
        feats["length"].append(len(fid))
        n_fs += int(use_fs)

    n = len(feats["input_ids"])
    lens = sorted(feats["length"])
    print(
        f"rows kept {n} (dropped {n_trunc} over {args.max_seq_len}, "
        f"{100*n_trunc/max(1,len(rows)):.2f}%); fewshot rows {n_fs}; "
        f"len p50={lens[n//2]} p99={lens[int(n*0.99)]} max={lens[-1]}; "
        f"total tokens {sum(lens)/1e6:.1f}M"
    )
    if args.dry_run:
        i = feats["length"].index(max(feats["length"]))
        print("---- longest example (decoded) ----")
        print(tok.decode(feats["input_ids"][i])[:1200])
        j = 0
        print("---- target of row 0 ----")
        print(repr(tok.decode([t for t in feats["labels"][j] if t != -100])))
        return

    ds = Dataset.from_dict(feats)

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or 0
        ids, labs, att = [], [], []
        for b in batch:
            k = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad] * k)
            labs.append(b["labels"] + [-100] * k)
            att.append([1] * len(b["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(att),
        }

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    for p_ in model.model.vision_tower.parameters():
        p_.requires_grad = False
    for p_ in model.model.multi_modal_projector.parameters():
        p_.requires_grad = False
    model.config.use_cache = False
    # A parent checkpoint saved by this script carries the greedy generation
    # config (do_sample=False, temperature=0.0) that vLLM needs. transformers
    # refuses to *re-save* that combination, which kills every checkpoint save
    # at the end of training, so reset it to a valid one here; the greedy JSON
    # is written back verbatim after the final save.
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid",
    )

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        accelerator_config={"split_batches": False},
    )
    sampler = TokenBudgetBatchSampler(feats["length"], args.max_tokens, args.seed)
    print(
        f"{len(sampler)} micro-batches/epoch, "
        f"{len(sampler)/args.grad_accum:.0f} optimizer steps/epoch"
    )

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=collate,
                num_workers=4,
                pin_memory=True,
            )

    trainer = BudgetTrainer(
        model=model, args=targs, train_dataset=ds, data_collator=collate
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # noqa: BLE001
        print("processor save failed:", e)
    for fn in (
        "preprocessor_config.json",
        "processor_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        src = os.path.join(args.model, fn)
        dst = os.path.join(final, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    # greedy decoding: vLLM reads generation_config.json as its sampling defaults
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(
            {
                "bos_token_id": 2,
                "eos_token_id": [1, 106],
                "pad_token_id": 0,
                "cache_implementation": "hybrid",
                "do_sample": False,
                "temperature": 0.0,
                "transformers_version": "4.57.3",
            },
            f,
            indent=2,
        )
    print("saved", final)


if __name__ == "__main__":
    main()
