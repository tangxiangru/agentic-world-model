#!/usr/bin/env python3
"""GRPO on GSM8K-train (+ augmented gsm8k) with an exact-match reward."""
from __future__ import annotations
import argparse, glob, json, os, random, re, shutil
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*([^\n]*)")


def norm_num(s):
    s = str(s).strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    try:
        f = float(s.strip())
    except (ValueError, OverflowError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
        return None
    return str(int(f)) if f == int(f) else str(round(f, 6))


def extract(text):
    ms = ANS_RE.findall(text)
    return norm_num(ms[-1]) if ms else None


def build_prompt(q):
    return ("<bos><start_of_turn>user\n" + MATH_PROMPT_TEMPLATE.format(prompt=q)
            + "<end_of_turn>\n<start_of_turn>model\n")


def reward_correct(completions, answer, **kw):
    return [1.0 if extract(c) == a else 0.0 for c, a in zip(completions, answer)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="runs/grpo1")
    ap.add_argument("--n-aug", type=int, default=15000)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--max-completion", type=int, default=400)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--vllm-util", type=float, default=0.28)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    rows, fewshot_pool = [], []
    for rec in gsm:
        g = norm_num(rec["answer"].split("####")[-1])
        if g is not None:
            rows.append({"prompt": build_prompt(rec["question"].strip()), "answer": g})
            fewshot_pool.append((rec["question"].strip(),
                                 rec["answer"].split("####")[0].strip() + "\n\nANSWER: " + g))
    if args.n_aug > 0:
        files = sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                                 "snapshots/*/data/*.parquet"))
        seen, aug = set(), []
        for f in files:
            df = pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
            df = df[df.problem_source == "augmented_gsm8k"]
            for prob, ans, _ in df.itertuples(index=False):
                if prob in seen or len(prob) > 1200:
                    continue
                a = norm_num(ans)
                if a is None:
                    continue
                seen.add(prob)
                aug.append({"prompt": build_prompt(prob.strip()), "answer": a})
            if len(aug) > args.n_aug * 2:
                break
        rng.shuffle(aug)
        rows += aug[:args.n_aug]
    # keep a fraction of prompts few-shot so RL does not over-specialise on 0-shot
    def make_prefix(k, exclude_q):
        picks = []
        while len(picks) < k:
            q, sol = fewshot_pool[rng.randrange(len(fewshot_pool))]
            if q == exclude_q:
                continue
            picks.append(f"{q}\n\nReasoning:\n{sol}")
        return "\n\n".join(picks) + "\n\n"

    P = "<bos><start_of_turn>user\n"
    for r in rows:
        if rng.random() < args.fewshot_frac:
            q = r["prompt"].split("Reasoning:")[0]
            body = r["prompt"][len(P):]
            r["prompt"] = P + make_prefix(rng.choice([2, 3, 4]), "") + body
    rng.shuffle(rows)
    ds = Dataset.from_list(rows)
    print("GRPO prompts:", len(ds), flush=True)

    from trl import GRPOConfig, GRPOTrainer
    from transformers import Gemma3ForConditionalGeneration
    from liger_kernel.transformers import apply_liger_kernel_to_gemma3
    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=False, cross_entropy=False)

    tok = AutoTokenizer.from_pretrained(args.init)
    cfg = AutoConfig.from_pretrained(args.init)
    Cls = (Gemma3ForConditionalGeneration
           if cfg.architectures[0] == "Gemma3ForConditionalGeneration"
           else __import__("transformers").AutoModelForCausalLM)
    model = Cls.from_pretrained(args.init, dtype=torch.bfloat16,
                                attn_implementation=os.environ.get("ATTN", "eager"))
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad = False

    conf = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_steps=args.steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.gens,
        generation_batch_size=args.bs * args.accum,
        max_prompt_length=1600,
        max_completion_length=args.max_completion,
        temperature=args.temp,
        top_p=0.95,
        top_k=64,
        beta=args.beta,
        loss_type="dr_grpo",
        scale_rewards="none",
        epsilon_high=0.28,
        num_iterations=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=0.5,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_util,
        vllm_max_model_length=2304,
        mask_truncated_completions=True,
        importance_sampling_level="token",
        log_completions=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        seed=args.seed,
    )
    tr = GRPOTrainer(model=model, reward_funcs=[reward_correct], args=conf,
                     train_dataset=ds, processing_class=tok)
    tr.train()
    final = os.path.join(args.out, "final")
    tr.save_model(final)
    tok.save_pretrained(final)
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    print("saved", final)


if __name__ == "__main__":
    main()
