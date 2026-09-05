#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt for GSM8K under the inspect harness.

Every training row is rendered with the *same* chat template the grader passes to
vLLM (templates/gemma3.jinja), the target ends with the terminator that template
closes an assistant turn with (<end_of_turn>, id 106, which is in the model's
generation_config eos_token_id), and loss is taken on completion tokens only.

Cross entropy is computed only at the positions that carry loss: Gemma3's vocab
is 262k, so materialising [B, T, V] logits for a 2.2k-token few-shot prompt is
what would OOM this run, not the weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response '
    'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    "answer to the problem.\n\n{prompt}\n\nRemember to put your answer on its own line "
    'at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    "answer to the problem, and you do not need to use a \\boxed command.\n\nReasoning:"
)
END_OF_TURN = "<end_of_turn>"


def eval_fewshots() -> list[str]:
    """The exact 10 shots the grader puts in the system message (gsm8k train, seed 42)."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    ds = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                    sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
    return [sample_to_fewshot(s) for s in ds]


class SFTData(Dataset):
    def __init__(self, rows, tok, template, shots, fewshot_frac, max_seq_len, seed):
        self.tok = tok
        self.examples = []
        rng = random.Random(seed)
        n_trunc = 0
        for r in rows:
            assert r["target"].endswith(END_OF_TURN), "target does not end with the stop token"

            user = PROMPT_TEMPLATE.format(prompt=r["question"])
            k = 0
            if rng.random() < fewshot_frac:
                k = rng.choice([2, 4, 6, 8, 10])
            msgs = []
            if k:
                msgs.append({"role": "system", "content": "\n\n".join(shots[:k])})
            msgs.append({"role": "user", "content": user})
            prompt = tok.apply_chat_template(msgs, chat_template=template,
                                             tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            t_ids = tok(r["target"], add_special_tokens=False).input_ids
            if len(p_ids) + len(t_ids) > max_seq_len:
                n_trunc += 1
                continue
            self.examples.append((p_ids, t_ids))
        self.n_trunc = n_trunc

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        p, t = self.examples[i]
        return {"input_ids": p + t, "labels": [-100] * len(p) + t}

    def lengths(self):
        return [len(p) + len(t) for p, t in self.examples]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        d = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * d)
        labels.append(b["labels"] + [-100] * d)
        mask.append([1] * len(b["input_ids"]) + [0] * d)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


class SparseCETrainer(Trainer):
    """Cross entropy over loss-carrying positions only (262k vocab)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.model if hasattr(model, "model") else model
        out = base(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        h = out.last_hidden_state[:, :-1, :]
        tgt = labels[:, 1:]
        sel = tgt != -100
        h_sel = h[sel]
        t_sel = tgt[sel]
        logits = model.lm_head(h_sel).float()
        loss = F.cross_entropy(logits, t_sel)
        return (loss, out) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit-rows", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    template = open(TEMPLATE).read()
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit_rows:
        rows = rows[: args.limit_rows]
    shots = eval_fewshots()

    ds = SFTData(rows, tok, template, shots, args.fewshot_frac, args.max_seq_len, args.seed)
    lens = ds.lengths()
    lens_sorted = sorted(lens)
    print(f"rows={len(ds)} dropped_too_long={ds.n_trunc} "
          f"tok p50={lens_sorted[len(lens)//2]} p99={lens_sorted[int(len(lens)*0.99)]} "
          f"max={lens_sorted[-1]} total_tokens={sum(lens)/1e6:.1f}M", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    assert getattr(model.config.text_config, "final_logit_softcapping", None) in (None, 0.0), \
        "final logit softcapping is on; the sparse-CE head would be wrong"
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 100000,
        save_total_limit=None,   # keep every checkpoint: epoch-boundary contrast is a lever
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=2,
        save_safetensors=True,
    )

    class LenDS(Dataset):
        """Trainer's length grouping reads a 'length' column off the dataset."""

        def __init__(self, inner):
            self.inner = inner
            self.length = inner.lengths()

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, i):
            return self.inner[i]

    from functools import partial

    trainer = SparseCETrainer(
        model=model,
        args=targs,
        train_dataset=LenDS(ds),
        data_collator=partial(collate, pad_id=tok.pad_token_id),
    )
    # our compute_loss returns a plain mean; let Trainer do the /accum itself
    trainer.model_accepts_loss_kwargs = False
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # vLLM loads gemma3 as a multimodal model: keep the processor files next to the weights
    import shutil
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAPSHOT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    # keep the parent's decode config untouched: decoding is a separate card
    shutil.copy(os.path.join(SNAPSHOT, "generation_config.json"),
                os.path.join(final, "generation_config.json"))
    print("saved", final)


if __name__ == "__main__":
    main()
