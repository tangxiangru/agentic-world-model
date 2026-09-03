#!/usr/bin/env python3
"""Reproduce, byte for byte, the 10-shot system message the grader builds.

inspect_evals/gsm8k builds it from the gsm8k TRAIN split (shuffle=True, seed=42,
limit=10) with sample_to_fewshot(). We import those exact call sites so the string
cannot drift from what the grader renders.
"""
from pathlib import Path

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

fewshots = hf_dataset(
    path="openai/gsm8k",
    data_dir="main",
    split="train",
    sample_fields=record_to_sample,
    shuffle=True,
    seed=42,
    limit=10,
)
text = "\n\n".join(sample_to_fewshot(s) for s in fewshots)
out = Path("/home/ben/task/data/fewshot_prefix.txt")
out.write_text(text)
print(f"wrote {out} ({len(text)} chars)")
print(text[:400])
print("...")
print(text[-300:])
