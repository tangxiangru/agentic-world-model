#!/usr/bin/env python3
"""Make a trainer checkpoint dir evaluable by vLLM: copy tokenizer files from the
base snapshot and write a deterministic generation_config that stops on
<end_of_turn> (106). Idempotent."""
import shutil
import sys
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

ckpt = Path(sys.argv[1])
tk = AutoTokenizer.from_pretrained(SNAP)
tk.save_pretrained(ckpt)
# Gemma3 is multimodal: vLLM needs the processor configs to start.
for fn in ("preprocessor_config.json", "processor_config.json"):
    src = Path(SNAP) / fn
    if src.exists():
        shutil.copy(src, ckpt / fn)
EOT = tk.convert_tokens_to_ids("<end_of_turn>")
gc = GenerationConfig(
    bos_token_id=tk.bos_token_id,
    eos_token_id=[tk.eos_token_id, EOT],
    pad_token_id=tk.pad_token_id,
    do_sample=False,
    cache_implementation="hybrid",
)
gc.save_pretrained(ckpt)
print(f"prepped {ckpt} (EOT={EOT})")
