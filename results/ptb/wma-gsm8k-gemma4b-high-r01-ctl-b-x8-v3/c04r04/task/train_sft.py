#!/usr/bin/env python3
"""Completion-only SFT of gemma-3-4b-pt (text tower) on GSM8K-style CoT data.

Prompts are rendered with the *grader's* chat template (templates/gemma3.jinja,
hash printed at every run) so training and grading see byte-identical strings.
Targets end with <end_of_turn>, the terminator that template stops on.
"""
import argparse, hashlib, json, os, random

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, set_seed)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"


def grader_fewshot() -> str:
    """The grader's 10-shot system message: GSM8K *train* split, shuffled seed 42, first 10.
    Same construction as inspect_evals.gsm8k.sample_to_fewshot."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(10))
    parts = []
    for r in ds:
        reasoning, target = r["answer"].split("####")
        parts.append(f"{r['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}")
    return "\n\n".join(parts)


def render_prompt(tok, question: str, fewshot: str | None = None) -> str:
    msgs = []
    if fewshot:
        msgs.append({"role": "system", "content": fewshot})
    msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class SFTData(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def build_examples(tok, data_path, max_len, eot_id, fewshot_pool=None, fewshot_frac=0.0, seed=0):
    rng = random.Random(seed)
    prompts, completions = [], []
    for line in open(data_path):
        r = json.loads(line)
        fs = None
        if fewshot_pool and rng.random() < fewshot_frac:
            if len(fewshot_pool) == 1:
                fs = fewshot_pool[0]
            else:
                k = rng.choice([1, 2, 3])
                fs = "\n\n".join(rng.sample(fewshot_pool, k))
        prompts.append(render_prompt(tok, r["question"], fs))
        # the target field is the exact string preflight checked, stop token included
        t = r["target"] if "target" in r else (r["solution"].strip() + "\nANSWER: " + r["answer"] + "<end_of_turn>")
        assert t.endswith("<end_of_turn>")
        completions.append(t)
    p_enc = tok(prompts, add_special_tokens=False)["input_ids"]
    c_enc = tok(completions, add_special_tokens=False)["input_ids"]
    examples, n_trunc = [], 0
    for p, c in zip(p_enc, c_enc):
        assert c[-1] == eot_id, c[-3:]
        if len(p) + len(c) > max_len:
            n_trunc += 1
            continue                      # drop rather than truncate: a cut target teaches no stop
        examples.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    print(f"examples: {len(examples)} kept, {n_trunc} dropped for exceeding max_len={max_len}", flush=True)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--fewshot-pool", type=str, default=None)
    ap.add_argument("--grader-prefix", action="store_true",
                    help="prepend the grader's exact 10-shot system message to every training prompt")
    ap.add_argument("--optim", type=str, default="adamw_bnb_8bit")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--attn", type=str, default="flash_attention_2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("grader template sha1:", hashlib.sha1(open(TEMPLATE_PATH, "rb").read()).hexdigest(), flush=True)
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(TEMPLATE_PATH).read()
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    assert isinstance(eot_id, int) and eot_id > 0, eot_id

    pool = None
    if args.grader_prefix:
        pool = [grader_fewshot()]
        args.fewshot_frac = 1.0
    elif args.fewshot_pool:
        pool = [json.loads(l)["text"] for l in open(args.fewshot_pool)]
    examples = build_examples(tok, args.data, args.max_len, eot_id, pool, args.fewshot_frac, args.seed)
    ds = SFTData(examples)

    if args.dry_run:
        import numpy as np
        L = [len(e["input_ids"]) for e in examples]
        nlab = [sum(1 for x in e["labels"] if x != -100) for e in examples[:2000]]
        print("token len p50/p90/p99/max:", int(np.percentile(L, 50)), int(np.percentile(L, 90)),
              int(np.percentile(L, 99)), max(L))
        print("label tokens p50:", int(np.percentile(nlab, 50)), "min:", min(nlab))
        ex = examples[0]
        print("=== decoded full ===")
        print(tok.decode(ex["input_ids"]))
        print("=== decoded labels only ===")
        print(tok.decode([t for t in ex["labels"] if t != -100]))
        print("=== last 6 label tokens ===",
              [tok.convert_ids_to_tokens(t) for t in ex["labels"] if t != -100][-6:])
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation=args.attn)
    model.config.use_cache = False
    # a parent packaged for greedy grading carries temperature 0 / do_sample false, which
    # transformers refuses to *save*; reset to a valid config (package_model.py rewrites it anyway)
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_k, gc.top_p = True, 1.0, 64, 0.95
    print(type(model).__name__, round(sum(p.numel() for p in model.parameters()) / 1e9, 2), "B params", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=20,
        save_steps=args.save_steps if args.save_steps else 10**9,
        save_strategy="steps" if args.save_steps else "no",
        save_total_limit=4,
        bf16=True,
        optim=args.optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    out = os.path.join(args.output_dir, "final")
    trainer.save_model(out)
    tok.save_pretrained(out)
    print("saved", out)


if __name__ == "__main__":
    main()
