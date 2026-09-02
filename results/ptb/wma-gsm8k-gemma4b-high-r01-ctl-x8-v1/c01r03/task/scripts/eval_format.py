"""Reproduce byte-for-byte the prompt the grader builds for inspect_evals/gsm8k.

The grader (evaluate.py) runs task `inspect_evals/gsm8k` with default args:
  fewshot=10, fewshot_seed=42, shuffle_fewshot=True
so the system message is 10 GSM8K *train* examples, and each user message is
MATH_PROMPT_TEMPLATE.format(prompt=question).

Rendering to text uses templates/gemma3.jinja (the file evaluate.py passes to vLLM).
"""

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import (
    MATH_PROMPT_TEMPLATE,
    record_to_sample,
    sample_to_fewshot,
    DATASET_PATH,
)

FEWSHOT_N = 10
FEWSHOT_SEED = 42


def build_system_message(n: int = FEWSHOT_N, seed: int = FEWSHOT_SEED) -> str:
    fewshots = hf_dataset(
        path=DATASET_PATH,
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return "\n\n".join([sample_to_fewshot(s) for s in fewshots])


def build_user_message(question: str) -> str:
    # inspect's prompt_template substitutes {prompt}
    return MATH_PROMPT_TEMPLATE.replace("{prompt}", question)


def fewshot_questions(n: int = FEWSHOT_N, seed: int = FEWSHOT_SEED) -> list[str]:
    fewshots = hf_dataset(
        path=DATASET_PATH,
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return [s.input for s in fewshots]


if __name__ == "__main__":
    sysmsg = build_system_message()
    print("=== SYSTEM MESSAGE ===")
    print(sysmsg)
    print("=== END (chars=%d) ===" % len(sysmsg))
    print()
    print("=== USER MESSAGE ===")
    print(build_user_message("Natalia sold clips to 48 friends."))
