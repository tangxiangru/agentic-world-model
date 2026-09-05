"""Byte-identical reproduction of the grader's prompt rendering.

The grader (evaluate.py -> inspect_evals/gsm8k) builds:
  system : 10 few-shot exemplars from the gsm8k TRAIN split (fewshot_seed=42, shuffled)
  user   : MATH_PROMPT_TEMPLATE.format(prompt=question)
and renders them with templates/gemma3.jinja through vLLM.

Everything here imports the grader's own code / template so training and grading
cannot drift (pitfall: template_unreachable).
"""
from __future__ import annotations

import functools
import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_source() -> str:
    with open(TEMPLATE_PATH, "r") as f:
        return f.read()


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@functools.lru_cache(maxsize=1)
def math_prompt_template() -> str:
    from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

    return MATH_PROMPT_TEMPLATE


@functools.lru_cache(maxsize=1)
def fewshot_system_message() -> str:
    """Exactly what inspect_evals.gsm8k puts in the system message (fewshot=10)."""
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
    return "\n\n".join([sample_to_fewshot(s) for s in fewshots])


def user_prompt(question: str) -> str:
    return math_prompt_template().replace("{prompt}", question)


@functools.lru_cache(maxsize=1)
def _jinja_template():
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template_source())


def render_prompt(question: str, fewshot: bool) -> str:
    """The exact string vLLM feeds the model, up to and including '<start_of_turn>model\\n'."""
    messages = []
    if fewshot:
        messages.append({"role": "system", "content": fewshot_system_message()})
    messages.append({"role": "user", "content": user_prompt(question)})
    return _jinja_template().render(
        messages=messages, bos_token="<bos>", add_generation_prompt=True
    )


def render_target(solution: str) -> str:
    """Assistant turn body + the terminator the grader stops on."""
    return solution.strip() + STOP_TOKEN


if __name__ == "__main__":
    print("template sha256:", template_sha256())
    sysmsg = fewshot_system_message()
    print("fewshot system message chars:", len(sysmsg))
    print("---- first 600 chars of system message ----")
    print(sysmsg[:600])
    print("---- rendered zero-shot prompt ----")
    print(render_prompt("What is 2+2?", fewshot=False))
    print("---- rendered target ----")
    print(render_target("2 + 2 = 4.\n\nANSWER: 4"))
