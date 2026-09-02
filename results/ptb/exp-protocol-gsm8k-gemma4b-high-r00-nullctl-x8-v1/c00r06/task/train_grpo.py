#!/usr/bin/env python3
"""GRPO on GSM8K train questions, starting from an SFT checkpoint."""
import argparse, json, os, re

import torch
from datasets import Dataset
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from gen import PROMPT_TEMPLATE, eval_fewshot_system
from inspect_ai.scorer._common import match_str

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")


def correctness_reward(completions, target, **kwargs):
    out = []
    for comp, t in zip(completions, target):
        txt = comp[0]["content"] if isinstance(comp, list) else comp
        try:
            ok = match_str(txt, t, location="end", ignore_case=True, numeric=True)[1]
        except Exception:
            ok = False
        out.append(1.0 if ok else 0.0)
    return out


ANSWER_RE = re.compile(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?\s*$")


def format_reward(completions, **kwargs):
    out = []
    for comp in completions:
        txt = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(0.05 if ANSWER_RE.search(txt.strip()) else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="ckpt/grpo1")
    ap.add_argument("--data", default="data/gsm8k_train_rest.jsonl")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=640)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--vllm-util", type=float, default=0.26)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--fewshot", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.init)
    ct = open("templates/gemma3.jinja").read()
    tok.chat_template = ct
    # gemma chat turns end with <end_of_turn> (106), not <eos> (1). TRL/vLLM use
    # tokenizer.eos_token_id for stop + truncation detection, so point it there.
    tok.eos_token = "<end_of_turn>"
    print("eos_token_id ->", tok.eos_token_id, flush=True)

    sysmsg = eval_fewshot_system()[0] if args.fewshot else None

    rows = [json.loads(l) for l in open(args.data)]
    recs = []
    for r in rows:
        msgs = []
        if sysmsg:
            msgs.append({"role": "system", "content": sysmsg})
        msgs.append({"role": "user",
                     "content": PROMPT_TEMPLATE.format(prompt=r["question"])})
        recs.append({"prompt": msgs,
                     "target": r["answer"].split("####")[-1].strip()})
    ds = Dataset.from_list(recs)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.config.use_cache = False
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_k = None
    model.generation_config.top_p = None
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=8,
        adam_beta2=0.99,
        max_grad_norm=0.5,
        weight_decay=0.0,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.gen_batch // args.micro_bs,
        generation_batch_size=args.gen_batch,
        num_generations=args.num_generations,
        num_iterations=1,
        beta=args.beta,
        temperature=args.temperature,
        top_p=1.0,
        max_prompt_length=2048,
        generation_kwargs={"stop_token_ids": [1, 106]},
        max_completion_length=args.max_completion,
        mask_truncated_completions=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_util,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        log_completions=False,
        seed=0,
    )

    trainer = GRPOTrainer(model=model, args=cfg,
                          reward_funcs=[correctness_reward, format_reward],
                          train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    import shutil
    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "cache_implementation": "hybrid",
                   "do_sample": False, "temperature": 0.0,
                   "transformers_version": "4.50.0.dev0"}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
