"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style targets.

Prompts are rendered with the grader's own chat template (scripts/fmt.py), so
the string the model is trained on is the string vLLM will feed it. Loss is
masked over the prompt; the target always ends in <end_of_turn>, the token the
grading template stops on.
"""
import argparse
import json
import os
import random
import shutil
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import render_prompt, render_target  # noqa: E402

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SFTRows(Dataset):
    def __init__(self, ids_list):
        self.rows = ids_list

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def token_budget_batches(lengths, budget, seed):
    """Length-sorted batches of roughly `budget` padded tokens each.

    Gemma-3's 262k vocabulary makes the fp32 logit tensor the memory bottleneck,
    so batches are capped by total padded tokens rather than by row count.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and m * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur, cur_max = cur + [i], m
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


class BudgetTrainer(Trainer):
    """Trainer whose train dataloader uses token-budget batches."""

    def __init__(self, *a, token_budget=8192, collate_fn=None, **kw):
        super().__init__(*a, **kw)
        self.token_budget = token_budget
        self._collate_fn = collate_fn

    def get_train_dataloader(self):
        ds = self.train_dataset
        lengths = [r["length"] for r in ds.rows]
        batches = token_budget_batches(lengths, self.token_budget, self.args.seed)
        print(f"token-budget batches: {len(batches)} "
              f"(median rows/batch {sorted(len(b) for b in batches)[len(batches)//2]})",
              flush=True)
        return DataLoader(ds, batch_sampler=batches, collate_fn=self._collate_fn,
                          num_workers=2, pin_memory=True)


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def build(tokenizer, path, max_seq_len, fewshot_frac, fewshot_system, seed, limit):
    rng = random.Random(seed)
    rows, truncated, dropped = [], 0, 0
    with open(path) as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            r = json.loads(line)
            sysmsg = fewshot_system if rng.random() < fewshot_frac else None
            prompt = render_prompt(tokenizer, r["question"], sysmsg)
            target = render_target(r["target"])
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            t_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(t_ids) > max_seq_len:
                truncated += 1
                if len(p_ids) >= max_seq_len - 16:
                    dropped += 1
                    continue
                t_ids = t_ids[: max_seq_len - len(p_ids)]
                if t_ids[-1] != tokenizer.convert_tokens_to_ids("<end_of_turn>"):
                    dropped += 1
                    continue
            rows.append({"input_ids": p_ids + t_ids,
                         "labels": [-100] * len(p_ids) + t_ids,
                         "length": len(p_ids) + len(t_ids)})
    lens = sorted(r["length"] for r in rows)
    print(f"rows={len(rows)} truncated={truncated} dropped={dropped} "
          f"p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}",
          flush=True)
    return SFTRows(rows)


def save_loadable(out_dir, base_model, tokenizer):
    """Write the files vLLM needs on top of trainer.save_model().

    Gemma-3 is a multimodal checkpoint, so vLLM builds a multimodal budget and
    reads the image-processor config even for a text-only prompt. Trainer does
    not copy those files, and their absence makes the engine exit 1 at startup
    (pitfall final_model_not_loadable).
    """
    tokenizer.save_pretrained(out_dir)
    # The grader reads sampling params from generation_config.json, so every
    # checkpoint ships greedy. Written here rather than through GenerationConfig,
    # which refuses to serialise do_sample=False together with a temperature.
    gen_path = os.path.join(out_dir, "generation_config.json")
    with open(gen_path) as f:
        gen = json.load(f)
    gen.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    with open(gen_path, "w") as f:
        json.dump(gen, f, indent=2)
    for name in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(base_model, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, name))
        else:
            print(f"WARNING: {name} missing from {base_model}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(os.path.join(TASK_DIR, "data", "eval_fewshot_system.txt")) as f:
        fewshot_system = f.read()

    ds = build(tokenizer, args.data, args.max_seq_len, args.fewshot_frac,
               fewshot_system, args.seed, args.limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False
    model.config.use_cache = False
    # A parent checkpoint produced by this script carries do_sample=False with
    # temperature 0.0, which GenerationConfig.validate(strict=True) rejects at save
    # time - it crashed exp-05 after a full 1.2 h of training. Reset to HF defaults
    # now; save_loadable() writes the greedy values back as plain JSON afterwards.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_k = 50
    model.generation_config.top_p = 1.0

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,  # unused: BudgetTrainer supplies its own sampler
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        max_grad_norm=1.0,
    )
    pad_id = tokenizer.pad_token_id or 0
    trainer = BudgetTrainer(model=model, args=targs, train_dataset=ds,
                            token_budget=args.token_budget,
                            collate_fn=lambda b: collate(b, pad_id))
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    save_loadable(final, args.model, tokenizer)
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
