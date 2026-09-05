import time, torch, sys
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

dtype = sys.argv[1]           # fp32 | bf16
gc = sys.argv[2] == "1"       # grad checkpointing
bs = int(sys.argv[3])
sl = int(sys.argv[4])
attn = sys.argv[5] if len(sys.argv) > 5 else "sdpa"

from liger_kernel.transformers import apply_liger_kernel_to_gemma3, apply_liger_kernel_to_gemma3_text
apply_liger_kernel_to_gemma3()
apply_liger_kernel_to_gemma3_text()

m = Gemma3ForConditionalGeneration.from_pretrained(
    BASE, dtype=torch.float32 if dtype == "fp32" else torch.bfloat16,
    attn_implementation=attn).cuda()
for n, p in m.named_parameters():
    if "vision_tower" in n or "multi_modal_projector" in n:
        p.requires_grad_(False)
m.config.use_cache = False
m.train()
if gc:
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit([p for p in m.parameters() if p.requires_grad], lr=1e-5)

ids = torch.randint(10, 200000, (bs, sl)).cuda()
lab = ids.clone()
amp = torch.autocast("cuda", torch.bfloat16) if dtype == "fp32" else torch.autocast("cuda", torch.bfloat16, enabled=False)
for i in range(6):
    if i == 2:
        torch.cuda.synchronize(); t0 = time.time()
    with amp:
        out = m(input_ids=ids, labels=lab)
    out.loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
dt = (time.time() - t0) / 4
print(f"{dtype} gc={gc} bs={bs} sl={sl} attn={attn}: {dt:.3f}s/step  {bs*sl/dt:.0f} tok/s  peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
