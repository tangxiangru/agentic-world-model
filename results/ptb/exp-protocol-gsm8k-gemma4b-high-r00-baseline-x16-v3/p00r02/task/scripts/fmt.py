"""Rendering helpers that reproduce the grader's prompt byte-for-byte.

Everything the trainer and the data builders need about *format* lives here so
there is exactly one definition of it.  The chat template is read from
templates/gemma3.jinja -- the same file evaluate.py hands to vLLM -- and its
sha256 is asserted, so a silent template swap cannot go unnoticed
(pitfall: template_unreachable).
"""

from __future__ import annotations

import hashlib
import os

from jinja2 import Environment
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE, copied verbatim.
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
EOT = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _env() -> Environment:
    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg: str) -> None:
        raise TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env


with open(TEMPLATE_PATH) as _f:
    _TEMPLATE = _env().from_string(_f.read())


def render_prompt(messages, add_generation_prompt: bool = True) -> str:
    """Render exactly as vLLM's chat endpoint does (bos comes from the template)."""
    return _TEMPLATE.render(
        messages=messages,
        bos_token=BOS,
        add_generation_prompt=add_generation_prompt,
    )


def user_message(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """One few-shot example, matching inspect_evals' sample_to_fewshot()."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def eval_fewshot_system() -> str:
    """The exact 10-shot system message the grader builds (gsm8k *train*, seed 42)."""
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
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def build_example(question: str, solution: str, system: str | None = None):
    """Return (prompt_text, target_text) for completion-only SFT.

    target_text ends with <end_of_turn>, the terminator of the grading template
    and a member of generation_config.eos_token_id (id 106).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message(question)})
    prompt = render_prompt(messages, add_generation_prompt=True)
    target = solution.strip() + EOT
    return prompt, target
