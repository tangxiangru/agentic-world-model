"""Rendering helpers shared by data building, training and offline probes.

Everything here reproduces, byte for byte, what the grader does:
  * the 10-shot system message inspect_evals/gsm8k builds (fewshot_seed=42),
  * the MATH_PROMPT_TEMPLATE it wraps each question in,
  * the templates/gemma3.jinja chat template evaluate.py hands to vLLM.

Nothing else in this repo is allowed to hand-roll a prompt string.
"""
from __future__ import annotations

import functools
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# copied from inspect_evals/gsm8k/gsm8k.py (checked by scripts/check_template.py)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


@functools.lru_cache(maxsize=1)
def _fewshots() -> tuple:
    from inspect_evals.gsm8k.gsm8k import DATASET_PATH, record_to_sample
    from inspect_ai.dataset import hf_dataset

    return tuple(
        hf_dataset(
            path=DATASET_PATH,
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            shuffle=True,
            seed=42,
            limit=10,
        )
    )


def fewshot_blocks() -> tuple:
    """The 10 rendered few-shot blocks, in the order the grader uses them."""
    from inspect_evals.gsm8k.gsm8k import sample_to_fewshot

    return tuple(sample_to_fewshot(s) for s in _fewshots())


def fewshot_system_message(n_shot: int = 10) -> str:
    """The exact system message inspect_evals/gsm8k puts in front of every item."""
    return "\n\n".join(fewshot_blocks()[:n_shot])


def fewshot_questions() -> tuple:
    return tuple(s.input for s in _fewshots())


@functools.lru_cache(maxsize=1)
def _template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def render_prompt(question: str, n_shot: int, tokenizer) -> str:
    """Render the graded prompt for `question`, ending in '<start_of_turn>model\\n'.

    n_shot=10 is exactly what the grader sends. Smaller values keep the first
    n of the same 10 examples so that the 10-shot string is a suffix-compatible
    extension of the shorter ones.
    """
    user = MATH_PROMPT_TEMPLATE.format(prompt=question)
    messages = []
    if n_shot > 0:
        messages.append({"role": "system", "content": fewshot_system_message(n_shot)})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages,
        chat_template=_template(),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_target(solution: str) -> str:
    """The assistant turn as the template would emit it: body + <end_of_turn>."""
    return solution.strip() + END_OF_TURN
