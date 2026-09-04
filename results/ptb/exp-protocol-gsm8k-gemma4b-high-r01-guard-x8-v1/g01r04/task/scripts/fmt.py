"""Prompt/target formatting that matches the grading harness byte-for-byte.

Everything the grader does is re-derived here from its own sources:
  * the chat template is read from templates/gemma3.jinja (the file evaluate.py
    hands to vLLM), never from the tokenizer's own copy;
  * the user prompt is inspect_evals.gsm8k.gsm8k.MATH_PROMPT_TEMPLATE;
  * the 10-shot system message is rebuilt with inspect's own hf_dataset call
    (train split, seed 42, shuffled) so it is the identical string.
"""

from __future__ import annotations

import functools
import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


@functools.lru_cache(maxsize=1)
def chat_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_sha() -> str:
    return hashlib.sha256(chat_template().encode()).hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_SNAPSHOT)


@functools.lru_cache(maxsize=1)
def user_template() -> str:
    from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

    return MATH_PROMPT_TEMPLATE


@functools.lru_cache(maxsize=1)
def fewshot_system() -> str:
    """The exact system message inspect builds for fewshot=10, seed=42."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import DATASET_PATH, record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path=DATASET_PATH,
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def user_message(question: str) -> str:
    return user_template().replace("{prompt}", question)


def render(question: str, completion: str | None, *, system: str | None) -> str:
    """Render a full conversation with the grader's template.

    completion=None renders the prompt with the generation prompt appended
    (exactly what vLLM is fed at eval time).
    """
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_message(question)})
    if completion is None:
        return tokenizer().apply_chat_template(
            msgs, chat_template=chat_template(), tokenize=False,
            add_generation_prompt=True,
        )
    msgs.append({"role": "assistant", "content": completion})
    return tokenizer().apply_chat_template(
        msgs, chat_template=chat_template(), tokenize=False,
        add_generation_prompt=False,
    )


def prompt_and_target(question: str, completion: str, *, system: str | None):
    """Return (prompt_text, target_text) whose concatenation is the rendered
    conversation. The target ends with the stop token the grader stops on."""
    prompt = render(question, None, system=system)
    full = render(question, completion, system=system)
    assert full.startswith(prompt), (prompt[-200:], full[len(prompt) - 200:][:400])
    target = full[len(prompt):]
    # template writes '<end_of_turn>\n' after the model turn; drop the trailing
    # newline so the sequence ends exactly on the stop token.
    assert target.endswith(STOP_TOKEN + "\n"), repr(target[-40:])
    return prompt, target[: -len("\n")]


if __name__ == "__main__":
    import sys

    print("template sha:", template_sha())
    sysmsg = fewshot_system()
    print("fewshot system chars:", len(sysmsg))
    print("fewshot system tokens:", len(tokenizer()(sysmsg)["input_ids"]))
    q = "Amy has 3 boxes with 7 pencils each. How many pencils does she have?"
    c = "Amy has 3 boxes of 7 pencils, so she has 3 * 7 = 21 pencils.\n\nANSWER: 21"
    for name, s in (("zeroshot", None), ("fewshot", sysmsg)):
        p, t = prompt_and_target(q, c, system=s)
        print("=" * 30, name)
        print("PROMPT>>>" + (p if s is None else p[:200] + " ...[cut]... " + p[-400:]) + "<<<")
        print("TARGET>>>" + t + "<<<")
        print("prompt tokens:", len(tokenizer()(p, add_special_tokens=False)["input_ids"]))
    sys.stdout.flush()
