"""Reproduce, byte-for-byte, the 10-shot system message that inspect_evals/gsm8k
builds at eval time (fewshot=10, fewshot_seed=42, shuffle_fewshot=True)."""
import os
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE

def build_fewshot_system() -> str:
    fewshots = hf_dataset(
        path="openai/gsm8k", data_dir="main", split="train",
        sample_fields=record_to_sample, shuffle=True, seed=42, limit=10,
    )
    return "\n\n".join([sample_to_fewshot(s) for s in fewshots])

if __name__ == "__main__":
    s = build_fewshot_system()
    with open("data/fewshot_system.txt", "w") as f:
        f.write(s)
    print(len(s), "chars")
    print(s[:1200])
    print("...")
    print("=== PROMPT TEMPLATE ===")
    print(MATH_PROMPT_TEMPLATE)
