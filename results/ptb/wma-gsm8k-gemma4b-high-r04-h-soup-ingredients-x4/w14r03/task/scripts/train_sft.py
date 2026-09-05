#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on the GSM8K-style corpus.

Prompt and target are rendered with templates/gemma3.jinja -- the exact file
evaluate.py hands to vLLM -- so training and grading see the same string.
Loss is on the assistant turn only, terminated by <end_of_turn>.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--parent", default=common.BASE_MODEL)
    p.add_argument("--max-rows", type=int, default=-1)
    p.add_argument("--fewshot-rows", type=int, default=1200,
                   help="rows rendered with the grader's exact 10-shot system prefix")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--fewshot-max-seq-len", type=int, default=3072)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--token-budget", type=int, default=40000,
                   help="max padded tokens per micro-batch")
    p.add_argument("--max-batch", type=int, default=96)
    p.add_argument("--greedy-gen-config", action="store_true",
                   help="save generation_config with temperature 0.0 (exp-03: +10.7 pp)")
    p.add_argument("--liger", action="store_true",
                   help="fused cross-entropy; matters because gemma3's vocab is 262144")
    p.add_argument("--dry-run", action="store_true",
                   help="tokenize + report length stats, render one example, no GPU")
    return p.parse_args()


def build_examples(args, tok):
    rows = [json.loads(l) for l in open(args.data)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    fewshot_sys = common.fewshot_system_message() if args.fewshot_rows else None
    n_fewshot = min(args.fewshot_rows, len(rows))

    examples, dropped, lens = [], 0, []
    for i, r in enumerate(rows):
        use_fs = i < n_fewshot
        prompt = common.render_prompt(
            tok, r["question"], system=fewshot_sys if use_fs else None
        )
        target = common.render_target(r["completion"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        cap = args.fewshot_max_seq_len if use_fs else args.max_seq_len
        if len(p_ids) + len(t_ids) > cap:
            dropped += 1
            continue
        examples.append(
            {
                "input_ids": p_ids + t_ids,
                "labels": [-100] * len(p_ids) + t_ids,
                "length": len(p_ids) + len(t_ids),
                "fewshot": use_fs,
            }
        )
        lens.append(len(p_ids) + len(t_ids))
    lens.sort()
    stats = {
        "n": len(examples),
        "dropped_too_long": dropped,
        "dropped_share": dropped / max(1, len(rows)),
        "p50": lens[len(lens) // 2],
        "p99": lens[int(0.99 * len(lens))],
        "max": lens[-1],
        "total_tokens": sum(lens),
        "n_fewshot": sum(e["fewshot"] for e in examples),
    }
    return examples, stats


def make_batches(lengths, budget, max_bs, seed):
    """Greedy token-budget batching.

    A fixed sequence count OOMs here: the corpus mixes ~330-token rows with
    ~2200-token few-shot-prefixed rows, and 48 x 3012 tokens does not fit
    (logs/smoke2.log). Bounding padded tokens per micro-batch instead keeps
    memory flat and wastes no compute on padding.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, curmax = [], [], 0
    for i in order:
        newmax = max(curmax, lengths[i])
        if cur and (newmax * (len(cur) + 1) > budget or len(cur) + 1 > max_bs):
            batches.append(cur)
            cur, curmax = [i], lengths[i]
        else:
            cur.append(i)
            curmax = newmax
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def main():
    args = parse_args()
    tok = common.get_tokenizer(args.parent)
    print("template sha256:", common.template_sha256(), flush=True)

    examples, stats = build_examples(args, tok)
    print("DATA STATS:", json.dumps(stats), flush=True)

    ex = examples[0]
    print("---- rendered example (labels unmasked part marked) ----", flush=True)
    print(repr(tok.decode(ex["input_ids"])[:600]), flush=True)
    tgt = [t for t in ex["labels"] if t != -100]
    print("TARGET:", repr(tok.decode(tgt)[-300:]), flush=True)
    assert tok.decode(tgt).endswith(common.STOP_TOKEN), "target must end with the stop token"

    if args.dry_run:
        json.dump(stats, open(os.path.join("/home/ben/task/analysis",
                                           "dryrun_" + os.path.basename(args.output_dir) + ".json"), "w"), indent=2)
        return

    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.parent, dtype=torch.float32, attn_implementation="flash_attention_2"
        )
    except Exception as e:  # pragma: no cover
        print("flash_attention_2 unavailable, falling back to sdpa:", repr(e)[:200], flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.parent, dtype=torch.float32, attn_implementation="sdpa"
        )

    # A greedy parent (temperature 0.0 + do_sample False) carries a
    # generation_config that GenerationConfig.save_pretrained REFUSES to write,
    # and PreTrainedModel.save_pretrained calls it with no try/except -- so the
    # run would die at save time after the whole epoch. The base snapshot's
    # config validates cleanly; the greedy JSON is written by hand after the
    # save anyway, so the served decode config is unaffected.
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig.from_pretrained(common.BASE_MODEL)

    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {frozen/1e6:.1f}M, trainable {trainable/1e6:.1f}M", flush=True)

    model.config.use_cache = False

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            return examples[i]

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=args.save_steps if args.save_steps else 10**9,
        save_strategy="steps" if args.save_steps else "no",
        save_total_limit=3,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        save_safetensors=True,
        use_liger_kernel=args.liger,
    )

    batches = make_batches(
        [e["length"] for e in examples], args.token_budget, args.max_batch, args.seed
    )
    bl = sorted(len(b) for b in batches)
    print(f"micro-batches: {len(batches)}, seqs/batch p50={bl[len(bl)//2]} "
          f"min={bl[0]} max={bl[-1]}", flush=True)

    class TokenBudgetTrainer(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader

            dl = DataLoader(
                self.train_dataset,
                batch_sampler=batches,
                collate_fn=self.data_collator,
                num_workers=4,
                pin_memory=True,
            )
            return self.accelerator.prepare(dl)

    trainer = TokenBudgetTrainer(
        model=model,
        args=targs,
        train_dataset=DS(),
        data_collator=Collator(tok.pad_token_id),
    )
    t0 = time.time()
    out = trainer.train()
    print("train_output:", out, flush=True)

    final = os.path.join(args.output_dir, "final")
    model.config.use_cache = True
    # Training keeps fp32 params (bf16 autocast), but a checkpoint whose
    # config.json says float32 is handed to vLLM with dtype='auto', and vLLM's
    # rule for a float32 config is to cast to float16 -- numerically unstable
    # for gemma3. Save in bfloat16, exactly like the base snapshot.
    model.to(torch.bfloat16)
    model.config.torch_dtype = torch.bfloat16
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = torch.bfloat16
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the multimodal side-car files so vLLM loads the checkpoint the same
    # way it loads the base snapshot
    import shutil
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.parent, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    # Trainer.save_model drops generation_config fields that match library
    # defaults and can collapse eos_token_id [1, 106] to 1, which would strip
    # <end_of_turn> as a stop token at grading time (pitfall eos_mismatch).
    # Restore the base snapshot's generation_config verbatim.
    shutil.copy(
        os.path.join(args.parent, "generation_config.json"),
        os.path.join(final, "generation_config.json"),
    )
    if args.greedy_gen_config:
        # exp-03: vLLM reads temperature/top_k/top_p out of this file whenever
        # the request omits them, and inspect always omits them. Greedy was
        # worth +10.7 pp on dev-150 over the base snapshot's sampling defaults.
        json.dump(
            {"bos_token_id": 2, "cache_implementation": "hybrid", "do_sample": False,
             "eos_token_id": [1, 106], "pad_token_id": 0, "temperature": 0.0,
             "top_k": 0, "top_p": 1.0, "transformers_version": "4.50.0.dev0"},
            open(os.path.join(final, "generation_config.json"), "w"), indent=2,
        )
    gc = json.load(open(os.path.join(final, "generation_config.json")))
    assert 106 in gc["eos_token_id"], gc
    cfg = json.load(open(os.path.join(final, "config.json")))
    if cfg.get("torch_dtype") != "bfloat16" or cfg.get("dtype") not in (None, "bfloat16"):
        cfg["torch_dtype"] = "bfloat16"
        cfg["dtype"] = "bfloat16"
        json.dump(cfg, open(os.path.join(final, "config.json"), "w"), indent=2)
    print("generation_config restored:", gc, flush=True)
    print("saved dtype:", cfg.get("dtype"), cfg.get("torch_dtype"), flush=True)
    json.dump(
        {"data_stats": stats, "wall_h": (time.time() - t0) / 3600,
         "final_loss": out.training_loss, "steps": out.global_step},
        open(os.path.join(args.output_dir, "train_summary.json"), "w"), indent=2,
    )
    print("saved ->", final, flush=True)


if __name__ == "__main__":
    main()
