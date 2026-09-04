"""Exact reproduction of the grader's prompt rendering.

The grader is `python evaluate.py`, which runs inspect_evals/gsm8k with
`chat_template=templates/gemma3.jinja` passed to vLLM.  Everything here is
derived from those two files, so training and grading render byte-identical
strings (pitfall: template_unreachable).
"""
from __future__ import annotations

import hashlib
import os

from jinja2 import Environment
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
# sha256 of templates/gemma3.jinja as shipped; guards against silent drift.
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

# inspect_evals/gsm8k/gsm8k.py :: MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
STOP = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_source() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def template_hash() -> str:
    return hashlib.sha256(template_source().encode()).hexdigest()


def _env() -> Environment:
    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = raise_exception
    return env


_TEMPLATE = None


def render_chat(messages, add_generation_prompt: bool = True) -> str:
    """Render exactly as vLLM does with --chat-template templates/gemma3.jinja."""
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _env().from_string(template_source())
    return _TEMPLATE.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=BOS,
    )


def user_message(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_system_message(n: int = 10, seed: int = 42) -> str:
    """Reproduce inspect_evals/gsm8k's 10-shot system message (fewshot_seed=42)."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def prompt_for(question: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message(question)})
    return render_chat(messages, add_generation_prompt=True)


if __name__ == "__main__":
    print("template sha256:", template_hash())
    sysmsg = fewshot_system_message()
    q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    zero = prompt_for(q)
    few = prompt_for(q, sysmsg)
    print("=" * 30, "ZERO-SHOT PROMPT", "=" * 30)
    print(repr(zero))
    print("=" * 30, "FEW-SHOT PROMPT (head/tail)", "=" * 30)
    print(repr(few[:600]))
    print("...")
    print(repr(few[-600:]))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    print("zero-shot tokens:", len(tok(zero, add_special_tokens=False).input_ids))
    print("few-shot tokens:", len(tok(few, add_special_tokens=False).input_ids))
