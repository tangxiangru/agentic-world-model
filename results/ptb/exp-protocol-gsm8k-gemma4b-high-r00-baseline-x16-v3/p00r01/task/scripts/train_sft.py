#!/usr/bin/env python3
"""SFT for gemma-3-4b-pt on GSM8K-style chain-of-thought.

Sequences are rendered with the *exact* chat template the grader uses
(templates/gemma3.jinja, hash-checked) and terminated with <end_of_turn>, the
token vllm stops on (generation_config.eos_token_id == [1, 106]).
Loss is taken on the completion only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

TASK_DIR = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates/gemma3.jinja")

# byte-for-byte copy of inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"
# distribution over the number of few-shot examples prepended as a system message
FEWSHOT_DIST = [(0, 0.70), (2, 0.10), (4, 0.08), (8, 0.06), (10, 0.06)]


def load_template() -> tuple[str, str]:
    raw = open(TEMPLATE_PATH, "rb").read()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()[:12]


class SFTRows(Dataset):
    def __init__(self, rows, tok, fewshot_pool, max_seq_len, seed=0, fewshot=True,
                 precompute=True):
        self.rows = rows
        self.tok = tok
        self.pool = fewshot_pool
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)
        self.fewshot = fewshot
        self.n_trunc = 0
        self.cache = None
        self.lengths = None
        if precompute:
            self.cache = [self._build(i) for i in range(len(rows))]
            self.lengths = [len(e["input_ids"]) for e in self.cache]

    def __len__(self):
        return len(self.rows)

    def _shots(self):
        if not self.fewshot:
            return 0
        r = self.rng.random()
        acc = 0.0
        for k, p in FEWSHOT_DIST:
            acc += p
            if r < acc:
                return k
        return 0

    def render(self, question: str, target: str, k_shots: int):
        msgs = []
        if k_shots:
            shots = self.rng.sample(self.pool, k_shots)
            msgs.append({"role": "system", "content": "\n\n".join(shots)})
        msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)})
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not target.endswith(END_OF_TURN):
            target += END_OF_TURN
        return prompt, target

    def _build(self, i):
        row = self.rows[i]
        k = self._shots()
        while True:
            prompt, completion = self.render(row["question"], row["target"], k)
            p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = self.tok(completion, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) <= self.max_seq_len or k == 0:
                break
            # never truncate the completion: shed few-shot examples instead
            k = k // 2
        ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + list(c_ids)
        if len(ids) > self.max_seq_len:
            ids, labels = ids[: self.max_seq_len], labels[: self.max_seq_len]
            self.n_trunc += 1
        return {"input_ids": ids, "labels": labels}

    def __getitem__(self, i):
        if self.cache is not None:
            return self.cache[i]
        return self._build(i)


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        d = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * d)
        labels.append(b["labels"] + [-100] * d)
        attn.append([1] * len(b["input_ids"]) + [0] * d)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


class LenGroupedTrainer(Trainer):
    """Length-grouped batching plus a completion-only loss head.

    Two reasons this is not the stock Trainer:
      * rows differ by 5x in length (a 10-shot prompt vs a zero-shot one), so
        without length grouping most of every batch is padding;
      * gemma-3's vocab is 262k, so materialising logits for a whole
        [16, 2048] batch costs 30 GB and OOMs an 80 GB H100. Only ~15% of the
        positions carry a label, so the lm_head is applied to those alone.
    """

    def _get_train_sampler(self, *a, **kw):
        from transformers.trainer_pt_utils import LengthGroupedSampler

        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=self.train_dataset.lengths,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.module if hasattr(model, "module") else model
        out = base.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        hidden = out.last_hidden_state[:, :-1, :]
        tgt = labels[:, 1:]
        sel = tgt != -100
        h = hidden[sel]                      # [N, H] - completion positions only
        y = tgt[sel]                         # [N]
        logits = base.lm_head(h).float()
        loss = torch.nn.functional.cross_entropy(
            logits, y, reduction="sum" if num_items_in_batch is not None else "mean"
        )
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        return (loss, out) if return_outputs else loss


def build_fewshot_pool(n=400, seed=42):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)
    pool = []
    for i in idx:
        r = ds[i]
        body, _, tail = r["answer"].rpartition("####")
        pool.append(f"{r['question']}\n\nReasoning:\n{body.strip()}\n\nANSWER: {tail.strip()}")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1792)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--no-fewshot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template, thash = load_template()
    print(f"chat template {TEMPLATE_PATH} sha256[:12]={thash}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = template

    rows = [json.loads(l) for l in open(args.data)]
    random.Random(args.seed).shuffle(rows)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    pool = build_fewshot_pool()
    ds = SFTRows(rows, tok, pool, args.max_seq_len, seed=args.seed,
                 fewshot=not args.no_fewshot)
    lens = sorted(ds.lengths)
    print(f"rows={len(ds)} tokens={sum(ds.lengths)/1e6:.1f}M p50={lens[len(lens)//2]} "
          f"p95={lens[int(len(lens)*.95)]} max={lens[-1]} "
          f"truncated={ds.n_trunc} ({ds.n_trunc/len(ds):.3%})")

    # ---- dry run: verify every row, never touch the GPU --------------------
    if args.dry_run:
        eot_id = tok.convert_tokens_to_ids(END_OF_TURN)
        for i in range(len(ds)):
            ex = ds[i]
            assert ex["input_ids"][-1] == eot_id, f"row {i} does not end with {END_OF_TURN}"
            assert ex["labels"][-1] == eot_id, f"row {i}: stop token carries no loss"
            assert sum(l != -100 for l in ex["labels"]) > 5, f"row {i} has no loss tokens"
        print(f"dry run ok: all {len(ds)} rows end in {END_OF_TURN} under loss")
        p, c = ds.render(rows[0]["question"], rows[0]["target"], 1)
        print("---- rendered example ----")
        print(p + c)
        print("---- end ----")
        return

    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
    )
    print(type(model).__name__, f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    # the vision tower is never exercised by text-only GSM8K prompts
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen vision params: {n_frozen/1e6:.0f}M")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = LenGroupedTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id or 0),
    )
    trainer.train()
    print(f"truncated rows during training: {ds.n_trunc}")

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    save_aux(args.model, final)
    print("saved", final)


def save_aux(src: str, dst: str) -> None:
    """Copy the files save_pretrained drops but vllm needs.

    gemma-3-4b-pt is a Gemma3ForConditionalGeneration checkpoint, so vllm builds
    an image processor even for text-only prompts; without
    preprocessor_config.json its engine dies with IndexError before loading any
    weights (pitfalls.yaml final_model_not_loadable).
    """
    import shutil

    for name in ("preprocessor_config.json", "processor_config.json"):
        s = os.path.join(src, name)
        if os.path.exists(s) and not os.path.exists(os.path.join(dst, name)):
            shutil.copy2(s, os.path.join(dst, name))
            print("copied", name)


if __name__ == "__main__":
    main()
