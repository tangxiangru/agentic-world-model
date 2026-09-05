#!/usr/bin/env python3
"""Same data, same collator, same loss as train_sft.py, but a hand-written loop:
isolates the Trainer/accelerate wrapper from the model + data path."""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sft import Collator, SFTRows, TokenBudgetBatches, _chunk_ce  # noqa: E402

M = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
tok = AutoTokenizer.from_pretrained(M)
ds = SFTRows("data/sft_v1.jsonl", tok, 3328, limit=400)
bs = TokenBudgetBatches(ds.lengths, 10000, 64, 0)
dl = DataLoader(ds, batch_sampler=bs, collate_fn=Collator(tok.pad_token_id),
                num_workers=int(os.environ.get("NW", 4)), pin_memory=True)
print("micro_batches", len(bs), flush=True)

model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.float32, attn_implementation="sdpa").cuda()
model.config.use_cache = False
for n, p in model.named_parameters():
    if n.startswith("model.vision_tower") or n.startswith("model.multi_modal_projector"):
        p.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-6)

ntok = 0
t0 = None
for i, batch in enumerate(dl):
    if i == 3:
        torch.cuda.synchronize()
        t0 = time.time()
        ntok = 0
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    labels = batch.pop("labels")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"], use_cache=False)[0]
        sh, sl = h[:, :-1, :], labels[:, 1:]
        sel = sl != -100
        loss = _chunk_ce(sh[sel], sl[sel], model.lm_head, 2048)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    ntok += batch["input_ids"].numel()
    if i == 12:
        break
torch.cuda.synchronize()
dt = time.time() - t0
print(f"manual loop: {ntok} tok in {dt:.2f}s -> {ntok/dt:.0f} tok/s, "
      f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
