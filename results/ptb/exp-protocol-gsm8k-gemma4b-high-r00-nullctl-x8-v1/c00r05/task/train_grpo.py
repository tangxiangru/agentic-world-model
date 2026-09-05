#!/usr/bin/env python3
import argparse, json, os, random, re
import torch
from datasets import Dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text):
    idx = text.rfind("ANSWER:")
    if idx == -1:
        ms = NUM_RE.findall(text)
        if not ms:
            return None
        s = ms[-1]
    else:
        m = NUM_RE.search(text[idx + 7:])
        if not m:
            return None
        s = m.group(0)
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def correctness_reward(completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        txt = c if isinstance(c, str) else c[0]["content"]
        p = extract_answer(txt)
        ok = p is not None and abs(p - a) < 1e-6
        fmt = 0.0 if "ANSWER:" in txt else -0.1
        out.append((1.0 if ok else 0.0) + fmt)
    return out


def fewshot_block(k=10, seed=42):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(k))
    parts = []
    for r in ds:
        p = r["answer"].split("####")
        parts.append(f"{r['question']}\n\nReasoning:\n" + "####".join(p[:-1]).strip()
                     + f"\n\nANSWER: {p[-1].strip()}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/rft_v1/final")
    ap.add_argument("--out", default="runs/grpo_v1")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--max-completion", type=int, default=448)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=60)
    ap.add_argument("--vllm-frac", type=float, default=0.22)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--data-seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    fsb = fewshot_block(10)
    rng = random.Random(args.data_seed)
    rows = []
    for r in gsm:
        a = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        u = MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip())
        if rng.random() < args.fewshot_frac:
            u = fsb + "\n\n" + u
        rows.append({"prompt": [{"role": "user", "content": u}], "answer": a})
    rng.shuffle(rows)
    ds = Dataset.from_list(rows)
    print("train prompts", len(ds), flush=True)

    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()
    tok.eos_token = "<end_of_turn>"  # the token the model actually stops on
    print("eos id", tok.eos_token_id, "pad id", tok.pad_token_id, flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad = False

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        generation_batch_size=args.gen_batch,
        num_generations=args.gens,
        max_completion_length=args.max_completion,
        max_prompt_length=1792,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        scale_rewards="group",
        loss_type="dapo",
        mask_truncated_completions=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_frac,
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=2560,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        max_grad_norm=0.5,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        save_only_model=True,
        report_to=[],
        log_completions=False,
    )

    trainer = GRPOTrainer(model=model, reward_funcs=correctness_reward,
                          args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    import shutil
    BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        shutil.copy(os.path.join(BASE, fn), os.path.join(final, fn))
    print("saved", final)


if __name__ == "__main__":
    main()
