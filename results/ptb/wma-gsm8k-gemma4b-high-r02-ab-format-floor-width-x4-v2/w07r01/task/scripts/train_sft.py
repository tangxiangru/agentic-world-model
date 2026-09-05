"""Completion-only SFT for gemma-3-4b-pt on pre-rendered prompt/completion jsonl.

The jsonl rows are already rendered through templates/gemma3.jinja (see build_sft.py),
so this script does no templating of its own: it tokenizes prompt+completion, masks
the prompt, and drops (never truncates) any row longer than --max-seq-len.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

SNAPSHOT = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
AUX_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
]


def build_dataset(path, tok, max_len, limit=None):
    ids_all, labels_all, lens = [], [], []
    n_drop = n_mismatch = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            full = tok(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"]
            if full[: len(p_ids)] != p_ids:
                n_mismatch += 1
                c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
                full = p_ids + c_ids
            if len(full) > max_len:
                n_drop += 1
                continue
            labels = [-100] * len(p_ids) + full[len(p_ids):]
            ids_all.append(full)
            labels_all.append(labels)
            lens.append(len(full))
    print(f"[data] kept={len(ids_all)} dropped_too_long={n_drop} tok_boundary_mismatch={n_mismatch}")
    print(f"[data] tokens={sum(lens):,} mean_len={sum(lens)/max(1,len(lens)):.1f} max_len={max(lens)}")
    return Dataset.from_dict({"input_ids": ids_all, "labels": labels_all, "length": lens})


class TokenBudgetSampler(torch.utils.data.Sampler):
    """Length-sorted batches capped by padded token count.

    Gemma-3's vocabulary is 262144, so the logits tensor is
    batch * seq * 262144; a fixed batch size OOMs on the long-sequence batches
    and wastes the GPU on the short ones. Cap batch*max_len instead.
    """

    def __init__(self, lengths, budget, max_bs, seed=0):
        self.lengths, self.budget, self.max_bs, self.seed = lengths, budget, max_bs, seed
        self.batches = self._build()

    def _build(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches, cur, cur_max = [], [], 0
        for i in order:
            m = max(cur_max, self.lengths[i])
            if cur and ((len(cur) + 1) * m > self.budget or len(cur) + 1 > self.max_bs):
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            batches.append(cur)
        return batches

    def set_epoch(self, epoch):
        g = torch.Generator().manual_seed(self.seed + epoch)
        perm = torch.randperm(len(self.batches), generator=g).tolist()
        self.order = [self.batches[i] for i in perm]

    def __iter__(self):
        if not hasattr(self, "order"):
            self.set_epoch(0)
        return iter(self.order)

    def __len__(self):
        return len(self.batches)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * n)
            lab.append(f["labels"] + [-100] * n)
            att.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }


class EpochSaver(TrainerCallback):
    """Keep an evaluable bf16 checkpoint at every epoch boundary."""

    def __init__(self, model, tok, out):
        self.model, self.tok, self.out = model, tok, out

    def on_epoch_end(self, args, state, control, **kw):
        ep = int(round(state.epoch or 0))
        if ep >= 1:
            save_full(self.model, self.tok, os.path.join(self.out, f"epoch{ep}"))


class BudgetTrainer(Trainer):
    """Trainer that batches by token budget instead of a fixed batch size."""

    def set_sampler(self, sampler, collator):
        self._sampler, self._collator = sampler, collator

    def get_train_dataloader(self):
        self._sampler.set_epoch(int(self.state.epoch or 0))
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=self._sampler,
            collate_fn=self._collator,
            num_workers=4,
            pin_memory=True,
        )


def save_full(model, tok, outdir):
    """Save a bf16 copy that is byte-compatible with how the grader loads the base.

    Training keeps fp32 master weights; the base snapshot is bfloat16 and vLLM reads
    config.torch_dtype, so saving fp32 would silently double the served model's memory
    and change its numerics relative to every measurement taken so far.
    """
    os.makedirs(outdir, exist_ok=True)
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
          for k, v in model.state_dict().items()}
    model.save_pretrained(outdir, safe_serialization=True, state_dict=sd)
    del sd
    cfg_path = os.path.join(outdir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    for key in ("torch_dtype", "dtype"):
        if key in cfg:
            cfg[key] = "bfloat16"
    cfg.setdefault("torch_dtype", "bfloat16")
    for sub in ("text_config", "vision_config"):
        if isinstance(cfg.get(sub), dict):
            for key in ("torch_dtype", "dtype"):
                if key in cfg[sub]:
                    cfg[sub][key] = "bfloat16"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    tok.save_pretrained(outdir)
    for name in AUX_FILES:
        src = os.path.join(SNAPSHOT, name)
        dst = os.path.join(outdir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print("[save]", outdir, sorted(os.path.basename(p) for p in glob.glob(outdir + "/*")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max padded tokens (batch*max_len) in one micro-batch")
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--ga", type=int, default=2)
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-each-epoch", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = build_dataset(args.data, tok, args.max_seq_len, args.limit)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.parent, dtype=torch.float32, attn_implementation="flash_attention_2"
        )
    except Exception as e:  # pragma: no cover
        print("[warn] flash_attention_2 unavailable:", e)
        model = AutoModelForCausalLM.from_pretrained(
            args.parent, dtype=torch.float32, attn_implementation="sdpa"
        )

    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable={n_train/1e9:.3f}B frozen={n_frozen/1e6:.1f}M")
    model.config.use_cache = False

    sampler = TokenBudgetSampler(ds["length"], args.token_budget, args.max_bs, args.seed)
    bs_hist = [len(b) for b in sampler.batches]
    print(f"[sampler] {len(sampler.batches)} micro-batches/epoch, "
          f"bs min/median/max = {min(bs_hist)}/{sorted(bs_hist)[len(bs_hist)//2]}/{max(bs_hist)}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,  # unused: batching is done by TokenBudgetSampler
        gradient_accumulation_steps=args.ga,
        use_liger_kernel=not args.no_liger,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    collator = Collator(tok.pad_token_id)
    trainer = BudgetTrainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.set_sampler(sampler, collator)
    if args.save_each_epoch:
        trainer.add_callback(EpochSaver(model, tok, args.out))
    out = trainer.train()
    print("[train]", out.metrics)
    save_full(model, tok, os.path.join(args.out, "final"))
    with open(os.path.join(args.out, "train_metrics.json"), "w") as f:
        json.dump({"metrics": out.metrics, "log": trainer.state.log_history[-50:]}, f, indent=2)


if __name__ == "__main__":
    main()
