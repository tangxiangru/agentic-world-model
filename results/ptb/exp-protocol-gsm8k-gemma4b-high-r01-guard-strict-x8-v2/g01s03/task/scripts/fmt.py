"""Exact rendering of the grader's prompt format.

The grader (evaluate.py -> inspect_evals/gsm8k) builds, for every item:
  system : 10 few-shot examples from the gsm8k TRAIN split (seed 42), joined by "\n\n"
  user   : MATH_PROMPT_TEMPLATE.format(prompt=question)
and renders the conversation with templates/gemma3.jinja, then generates until
<end_of_turn> (id 106, in the base generation_config's eos_token_id list).

Everything here reads the same objects the grader reads, so training and
grading cannot drift apart (pitfall: template_unreachable).
"""

import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
STOP_TOKEN = "<end_of_turn>"

from inspect_evals.gsm8k.gsm8k import (  # noqa: E402
    MATH_PROMPT_TEMPLATE,
    record_to_sample,
    sample_to_fewshot,
)


def chat_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha() -> str:
    return hashlib.sha256(chat_template().encode()).hexdigest()[:12]


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


_FEWSHOT_CACHE = {}


def fewshot_system_message(n: int = 10, seed: int = 42) -> str:
    """Byte-identical to what inspect's gsm8k task puts in the system turn."""
    key = (n, seed)
    if key not in _FEWSHOT_CACHE:
        from inspect_ai.dataset import hf_dataset

        fewshots = hf_dataset(
            path="openai/gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            shuffle=True,
            seed=seed,
            limit=n,
        )
        _FEWSHOT_CACHE[key] = "\n\n".join(sample_to_fewshot(s) for s in fewshots)
    return _FEWSHOT_CACHE[key]


def get_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)
    tok.chat_template = chat_template()
    return tok


def render_prompt(tok, question: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_prompt(question)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render_target(solution: str, answer: str) -> str:
    """Assistant turn, ending on the token vLLM stops at."""
    body = solution.strip()
    return f"{body}\n\nANSWER: {answer}{STOP_TOKEN}"
