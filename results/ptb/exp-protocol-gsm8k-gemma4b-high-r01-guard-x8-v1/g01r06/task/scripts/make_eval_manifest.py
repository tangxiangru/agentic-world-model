"""Record which benchmark items --limit 200 scores, as opaque ids only.

Rule 7: the test copy is input to the contamination checker only. This writes
no question or answer text - just the stable ids - so the manifest can be
committed next to the cards without becoming a training-data hazard.
"""
import json
from inspect_evals.gsm8k.gsm8k import record_to_sample
from inspect_ai.dataset import hf_dataset

ds = hf_dataset(path="openai/gsm8k", data_dir="main", split="test",
                sample_fields=record_to_sample)
with open("/home/ben/task/data/eval_manifest.jsonl", "w") as f:
    for s in list(ds)[:200]:
        f.write(json.dumps({"id": s.id}) + "\n")
print("wrote 200 ids")
