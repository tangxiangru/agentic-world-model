"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style math CoT.

Prompts are rendered with templates/gemma3.jinja -- the exact file the grader
passes to vLLM -- so training and grading see byte-identical strings
(pitfall: template_unreachable).  Loss is on the completion only; every target
already ends with <end_of_turn>, the terminator that template stops on and that
is in generation_config.eos_token_id (pitfall: eos_mismatch).

Only the text tower is trained: Gemma3ForCausalLM reads the 444 language_model.*
tensors out of the multimodal checkpoint (verified key-by-key) and saves a
gemma3_text directory that vLLM and evaluate.py both accept.

Cross-entropy over the 262k-token vocabulary is computed in re-materialised
chunks over the completion tokens only; the stock path builds a
[batch, seq, 262144] fp32 tensor and OOMs an 80 GB H100 at batch 16.
"""
import argparse, json, os, random, shutil
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint as ckpt_fn
from transformers import (AutoTokenizer, Gemma3ForCausalLM, Trainer,
                          TrainingArguments, set_seed)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TPL = "/home/ben/task/templates/gemma3.jinja"
STOP = "<end_of_turn>"


def build_examples(tok, tpl, rows, max_len):
    """Tokenise into input_ids + labels with completion-only loss."""
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                chat_template=tpl, tokenize=False,
                                add_generation_prompt=True)
        for r in rows
    ]
    for r in rows:
        assert r["target"].endswith(STOP), "target does not end with the grader's stop token"
    targets = [r["target"] + "\n" for r in rows]
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    t_ids = tok(targets, add_special_tokens=False)["input_ids"]
    out, dropped = [], 0
    for p, t in zip(p_ids, t_ids):
        if len(p) + len(t) > max_len:
            dropped += 1
            continue
        out.append({"input_ids": p + t, "labels": [-100] * len(p) + t,
                    "length": len(p) + len(t)})
    return out, dropped


@dataclass
class Collator:
    pad_id: int

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, ex):
        self.ex = ex

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


CHUNK = 4096


def _chunk_ce(head_w, h, y):
    return F.cross_entropy(F.linear(h, head_w).float(), y, reduction="sum")


class ChunkedCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        core = self.accelerator.unwrap_model(model)
        hs = model.model(input_ids=inputs["input_ids"],
                         attention_mask=inputs["attention_mask"]).last_hidden_state
        h = hs[:, :-1, :].reshape(-1, hs.size(-1))
        y = labels[:, 1:].reshape(-1).to(h.device)
        keep = y != -100
        h, y = h[keep], y[keep]
        n = y.numel()
        w = core.lm_head.weight
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, n, CHUNK):
            total = total + ckpt_fn(_chunk_ce, w, h[i:i + CHUNK], y[i:i + CHUNK],
                                    use_reentrant=False)
        loss = total / max(n, 1)
        return (loss, None) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.parent)
    tpl = open(TPL).read()

    rows = [json.loads(l) for l in open(args.data)]
    random.Random(args.seed).shuffle(rows)
    if args.n:
        rows = rows[: args.n]
    ex, dropped = build_examples(tok, tpl, rows, args.max_seq_len)
    lens = np.array([e["length"] for e in ex])
    print(f"rows={len(rows)} kept={len(ex)} dropped_too_long={dropped} "
          f"({dropped/max(1,len(rows)):.3%}) len p50={int(np.percentile(lens,50))} "
          f"p99={int(np.percentile(lens,99))} max={lens.max()} "
          f"total_tokens={lens.sum()/1e6:.1f}M", flush=True)
    assert dropped / max(1, len(rows)) < 0.02, "over 2% of rows truncate"

    model = Gemma3ForCausalLM.from_pretrained(args.parent, dtype=torch.bfloat16,
                                              attn_implementation=os.environ.get("ATTN","flash_attention_2"))
    model.config.use_cache = False
    print(f"trainable={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
        optim=args.optim,
        save_safetensors=True,
    )

    trainer = ChunkedCETrainer(model=model, args=targs, train_dataset=ListDataset(ex),
                               data_collator=Collator(tok.pad_token_id))
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
