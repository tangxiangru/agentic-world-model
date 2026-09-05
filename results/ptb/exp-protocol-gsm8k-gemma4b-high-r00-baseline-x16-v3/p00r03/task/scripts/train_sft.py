"""SFT for gemma-3-4b-pt on GSM8K-style CoT, rendered with the grader's own template.

Prompt/completion strings are built with the *exact* markers templates/gemma3.jinja
emits, so training and grading see byte-identical renderings.  Loss is computed on
the completion only.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_SNAPSHOT, TASK_DIR, read_jsonl  # noqa: E402

from transformers import (  # noqa: E402
    AutoTokenizer,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"


def render_prompt(user_text: str, system_text: str | None = None) -> str:
    """Byte-for-byte what templates/gemma3.jinja produces up to the model turn."""
    first = (system_text.strip() + "\n\n") if system_text else ""
    return f"{BOS}{SOT}user\n{first}{user_text.strip()}{EOT}\n{SOT}model\n"


def render_completion(target: str) -> str:
    """The trained target ends on token 106 (<end_of_turn>) and nothing after it.

    Data files carry the stop token explicitly so preflight can verify it; this
    only appends it when a file predates that convention.
    """
    t = target.strip()
    return t if t.endswith(EOT) else t + EOT


class SFTData(Dataset):
    def __init__(self, rows, tok, max_len, fewshot_prob=0.0, fewshot_pool=None, seed=0):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len
        self.fewshot_prob = fewshot_prob
        self.fewshot_pool = fewshot_pool or []
        self.rng = random.Random(seed)
        self.cache = None
        if fewshot_prob <= 0:  # deterministic rendering -> tokenize once, up front
            ps = [render_prompt(r["prompt"]) for r in rows]
            cs = [render_completion(r["completion"]) for r in rows]
            pid = tok(ps, add_special_tokens=False)["input_ids"]
            cid = tok(cs, add_special_tokens=False)["input_ids"]
            self.cache = [self._pack(p, c) for p, c in zip(pid, cid)]

    def _pack(self, pid, cid):
        ids = list(pid) + list(cid)
        labels = [-100] * len(pid) + list(cid)
        if len(ids) > self.max_len:  # keep the completion, trim the prompt head
            over = len(ids) - self.max_len
            ids = [ids[0]] + ids[1 + over:]
            labels = [-100] + labels[1 + over:]
        return {"input_ids": ids, "labels": labels}

    def __len__(self):
        return len(self.rows)

    def _system(self):
        if not self.fewshot_pool or self.rng.random() >= self.fewshot_prob:
            return None
        k = self.rng.choice([2, 4, 8, 10])
        shots = self.rng.sample(self.fewshot_pool, min(k, len(self.fewshot_pool)))
        return "\n\n".join(shots)

    def __getitem__(self, i):
        if self.cache is not None:
            return self.cache[i]
        r = self.rows[i]
        p = render_prompt(r["prompt"], self._system())
        c = render_completion(r["completion"])
        pid = self.tok(p, add_special_tokens=False)["input_ids"]
        cid = self.tok(c, add_special_tokens=False)["input_ids"]
        return self._pack(pid, cid)


class TokenBudgetBatches:
    """Length-bucketed batches with a fixed token budget.

    Gemma-3's 262k vocabulary makes the loss logits (bf16 + an fp32 copy) the
    dominant memory term, so peak memory tracks tokens-per-microbatch, not rows.
    Capping padded tokens keeps peak flat and keeps padding waste low.
    """

    def __init__(self, lengths, budget, seed=0, max_rows=64):
        self.lengths = lengths
        self.budget = budget
        self.seed = seed
        self.max_rows = max_rows
        self.epoch = 0
        self._batches = self._build(0)

    def _build(self, epoch):
        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        # sort inside large megabatches so ordering stays stochastic across epochs
        mega = 4096
        batches = []
        for s in range(0, len(idx), mega):
            chunk = sorted(idx[s: s + mega], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                m = max(cur_max, self.lengths[i])
                if cur and (m * (len(cur) + 1) > self.budget or len(cur) >= self.max_rows):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur.append(i)
                    cur_max = m
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch = epoch
            self._batches = self._build(epoch)

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, mask = [], [], []
        for f in feats:
            d = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * d)
            labels.append(f["labels"] + [-100] * d)
            mask.append([1] * len(f["input_ids"]) + [0] * d)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=BASE_SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--fewshot-prob", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--token-budget", type=int, default=10240)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--group-by-length", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = read_jsonl(args.data)
    if args.max_rows:
        rows = rows[: args.max_rows]

    fewshot_pool = None
    if args.fewshot_prob > 0:
        import json as _json

        pool_path = os.path.join(TASK_DIR, "data", "fewshot_pool.json")
        with open(pool_path) as f:
            fewshot_pool = _json.load(f)

    ds = SFTData(rows, tok, args.max_len, args.fewshot_prob, fewshot_pool, args.seed)

    if args.dry_run:
        ex = ds[0]
        print("=== rendered example ===")
        print(tok.decode(ex["input_ids"]))
        print("=== loss tokens ===")
        print(tok.decode([t for t in ex["labels"] if t != -100]))
        lens = [len(ds[i]["input_ids"]) for i in range(0, len(ds), max(1, len(ds) // 3000))]
        lens.sort()
        print(
            json.dumps(
                {
                    "n_rows": len(ds),
                    "sampled": len(lens),
                    "p50": lens[len(lens) // 2],
                    "p95": lens[int(len(lens) * 0.95)],
                    "p99": lens[int(len(lens) * 0.99)],
                    "max": lens[-1],
                    "frac_truncated": sum(l >= args.max_len for l in lens) / len(lens),
                },
                indent=2,
            )
        )
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # text-only task: freeze the vision stack
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") \
           or name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen params: {n_frozen/1e6:.1f}M; trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
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
        save_total_limit=3,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        group_by_length=args.group_by_length,
        max_grad_norm=1.0,
        save_safetensors=True,
    )

    lengths = [len(ds[i]["input_ids"]) for i in range(len(ds))]
    batch_sampler = TokenBudgetBatches(lengths, args.token_budget, seed=args.seed)
    print(
        f"token-budget batching: {len(batch_sampler)} microbatches/epoch, "
        f"mean rows/batch {len(ds)/max(1,len(batch_sampler)):.1f}"
    )

    class BucketTrainer(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader

            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=2,
                pin_memory=True,
            )

    trainer = BucketTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    model.config.use_cache = True
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # pragma: no cover
        print("processor save failed:", e)
    for fn in ["preprocessor_config.json", "processor_config.json", "added_tokens.json",
               "special_tokens_map.json", "tokenizer.model", "generation_config.json"]:
        src = os.path.join(args.model, fn)
        dst = os.path.join(final, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
    print("saved", final)


if __name__ == "__main__":
    main()
