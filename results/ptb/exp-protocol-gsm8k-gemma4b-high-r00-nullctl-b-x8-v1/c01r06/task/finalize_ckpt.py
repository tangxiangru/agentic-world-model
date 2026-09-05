"""Copy tokenizer/processor aux files into a bare Trainer checkpoint so vLLM can load it."""
import os
import shutil
import sys

SNAP = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
FILES = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
         "added_tokens.json", "special_tokens_map.json",
         "preprocessor_config.json", "processor_config.json",
         "generation_config.json"]

dst = sys.argv[1]
for fn in FILES:
    src = os.path.join(SNAP, fn)
    if os.path.exists(src) and not os.path.exists(os.path.join(dst, fn)):
        shutil.copy(src, os.path.join(dst, fn))
print("finalized", dst, sorted(os.listdir(dst)))
