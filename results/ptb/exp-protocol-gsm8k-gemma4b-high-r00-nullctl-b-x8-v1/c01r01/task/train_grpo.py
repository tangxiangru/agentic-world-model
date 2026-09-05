#!/usr/bin/env python3
"""GRPO (verifiable-reward RL) on GSM8K *train* prompts, starting from the SFT checkpoint."""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.MULTILINE)


def norm(s):
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


def extract(text):
    m = ANS_RE.findall(text)
    return norm(m[-1]) if m else None


def correctness_reward(completions, answer, **kwargs):
    out = []
    for c, gold in zip(completions, answer):
        txt = c[0]["content"] if isinstance(c, list) else c
        pred = extract(txt)
        out.append(1.0 if (pred is not None and pred == gold) else 0.0)
    return out


def format_reward(completions, **kwargs):
    """Small shaping term: response must end with a well-formed ANSWER line."""
    out = []
    for c in completions:
        txt = (c[0]["content"] if isinstance(c, list) else c).strip()
        lines = [l for l in txt.split("\n") if l.strip()]
        ok = bool(lines) and bool(re.fullmatch(r"ANSWER:\s*\S+", lines[-1].strip()))
        out.append(0.1 if ok else 0.0)
    return out


def build_prompts(fewshot_prob, seed, n_omi=0):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for row in ds.select(range(400)):
        reasoning, _, final = row["answer"].partition("####")
        reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
        pool.append(f"{row['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {final.strip()}")

    items = [(r["question"], norm(r["answer"].split("####")[-1])) for r in ds]
    if n_omi > 0:
        import glob

        import pyarrow.parquet as pq

        seen = {q for q, _ in items}
        extra = []
        for f in sorted(glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
        )):
            t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pydict()
            for p, a, s in zip(t["problem"], t["expected_answer"], t["problem_source"]):
                if s == "augmented_gsm8k" and p not in seen:
                    seen.add(p)
                    extra.append((p, norm(a)))
            if len(extra) >= n_omi * 2:
                break
        random.Random(seed).shuffle(extra)
        items += extra[:n_omi]

    # A small pool of *fixed* k-shot prefixes so vLLM prefix-caching stays effective,
    # while still matching the 10-shot prompt shape the harness uses at eval time.
    prng = random.Random(seed + 99)
    prefixes = ["\n\n".join(prng.sample(pool, k)) + "\n\n" for k in (10, 10, 10, 5, 5, 3) for _ in range(1)]

    rng = random.Random(seed)
    rows = []
    for q, a in items:
        user = MATH_PROMPT_TEMPLATE.format(prompt=q.strip())
        if rng.random() < fewshot_prob:
            user = rng.choice(prefixes) + user
        rows.append({"prompt": [{"role": "user", "content": user}], "answer": a})
    rng.shuffle(rows)
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="work/grpo_v1")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--fewshot-prob", type=float, default=0.25)
    ap.add_argument("--n-omi", type=int, default=8000)
    ap.add_argument("--vllm-util", type=float, default=0.2)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from liger_kernel.transformers import apply_liger_kernel_to_gemma3

    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=False)

    tok = AutoTokenizer.from_pretrained(args.init)
    tok.chat_template = open("templates/gemma3.jinja").read()
    tok.padding_side = "left"
    # The model ends its turn with <end_of_turn> (106), not <eos> (1). TRL decides
    # "was this completion truncated?" by comparing the last token to
    # processing_class.eos_token_id, so without this every completion looks clipped
    # and mask_truncated_completions zeroes out the entire batch.
    tok.eos_token = "<end_of_turn>"
    assert tok.eos_token_id == 106, tok.eos_token_id

    ds = build_prompts(args.fewshot_prob, args.seed, args.n_omi)
    print(f"{len(ds)} RL prompts", flush=True)
    print("--- sample rendered prompt ---")
    print(repr(tok.apply_chat_template(ds[0]["prompt"], tokenize=False, add_generation_prompt=True))[:600])

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_grad_norm=0.2,
        adam_beta2=0.95,
        weight_decay=0.0,
        bf16=True,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        top_p=1.0,
        beta=args.beta,
        loss_type="dr_grpo",
        scale_rewards="none",
        mask_truncated_completions=True,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_util,
        vllm_max_model_length=args.max_prompt + args.max_completion + 16,
        report_to=[],
        seed=args.seed,
        log_completions=False,
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "eager"},
    )

    trainer = GRPOTrainer(
        model=args.init,
        reward_funcs=[correctness_reward, format_reward],
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok_out = AutoTokenizer.from_pretrained(args.init)
    tok_out.save_pretrained(args.out)
    for fn in ["generation_config.json", "preprocessor_config.json", "processor_config.json"]:
        src, dst = os.path.join(args.init, fn), os.path.join(args.out, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print("done", flush=True)


if __name__ == "__main__":
    main()
