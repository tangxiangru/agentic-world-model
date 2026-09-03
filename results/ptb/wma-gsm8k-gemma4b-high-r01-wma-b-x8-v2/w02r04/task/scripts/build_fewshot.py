#!/usr/bin/env python3
"""Reproduce the grader's 10-shot system message byte-for-byte.

inspect_evals.gsm8k builds it from openai/gsm8k *train* with shuffle=True,
seed=42, limit=10; we call the same functions so it cannot drift.
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def fewshot_system_message(fewshot: int = 10, fewshot_seed: int = 42) -> str:
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import DATASET_PATH, record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path=DATASET_PATH,
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=fewshot_seed,
        limit=fewshot,
    )
    return "\n\n".join([sample_to_fewshot(s) for s in fewshots])


if __name__ == "__main__":
    m = fewshot_system_message()
    print(m)
    print("---- chars:", len(m))
