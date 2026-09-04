#!/usr/bin/env python3
"""Where does the training step time go? Fixed synthetic batch, several configs."""
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

M = os.environ["PTB_BASE_MODEL_SNAPSHOT"]


def run(dtype, attn, ckpt, bs, seqlen, autocast, steps=4):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(M, dtype=dtype, attn_implementation=attn).cuda()
    model.config.use_cache = False
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or n.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ids = torch.randint(100, 200000, (bs, seqlen), device="cuda")
    am = torch.ones_like(ids)
    if os.environ.get("PAD"):
        am[: bs // 2, -seqlen // 4 :] = 0
    lab = ids.clone()
    t0 = None
    for i in range(steps):
        if i == 1:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            h = model.model(input_ids=ids, attention_mask=am, use_cache=False)[0]
            sel = h[:, :-1, :].reshape(-1, h.size(-1))[:bs * 200]
            tgt = lab[:, 1:].reshape(-1)[: bs * 200]
            if os.environ.get("CHUNK"):
                import torch.utils.checkpoint as cp
                tot = sel.new_zeros((), dtype=torch.float32)
                C = 2048
                for i in range(0, sel.size(0), C):
                    tot = tot + cp.checkpoint(
                        lambda h, l: F.cross_entropy(model.lm_head(h).float(), l, reduction="sum"),
                        sel[i:i+C], tgt[i:i+C], use_reentrant=False)
                loss = tot / sel.size(0)
            else:
                lg = model.lm_head(sel).float()
                loss = F.cross_entropy(lg, tgt)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / (steps - 1)
    tok = bs * seqlen
    print(
        f"dtype={str(dtype):>14} attn={attn:<18} ckpt={int(ckpt)} autocast={int(autocast)} "
        f"bs={bs} T={seqlen} -> {dt:.3f}s/step  {tok/dt:8.0f} tok/s  "
        f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run(torch.float32, "sdpa", True, 24, 400, True)
