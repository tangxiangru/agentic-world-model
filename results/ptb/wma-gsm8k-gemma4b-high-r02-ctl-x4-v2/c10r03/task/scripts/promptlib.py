"""Reproduce the exact prompt rendering that evaluate.py -> inspect_evals/gsm8k uses.

Two things must agree between training and grading:
  * the text of the user turn (MATH_PROMPT_TEMPLATE),
  * the chat template (templates/gemma3.jinja), whose terminator is <end_of_turn>.

The grader is inspect_ai `match(location="end", numeric=True)`: it takes the LAST
numeric token of the completion and compares it to the gold answer. So the final
line "ANSWER: N" is what gets read, and nothing numeric may follow it.
"""

from __future__ import annotations

import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_system_message(fewshot: int = 10, fewshot_seed: int = 42) -> str:
    """Exactly what inspect_evals.gsm8k builds for its system_message() solver."""
    from inspect_ai.dataset import hf_dataset

    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=fewshot_seed,
        limit=fewshot,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def load_chat_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def render(tokenizer, question: str, system: str | None, completion: str | None):
    """Render prompt (and optionally prompt+completion) with the grader's template.

    Returns (prompt_text, full_text). prompt_text ends with '<start_of_turn>model\\n'.
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_prompt(question)})
    tpl = load_chat_template()
    prompt_text = tokenizer.apply_chat_template(
        msgs, chat_template=tpl, tokenize=False, add_generation_prompt=True
    )
    full_text = None
    if completion is not None:
        c = completion.strip()
        if not c.endswith(STOP_TOKEN):
            c += STOP_TOKEN
        full_text = prompt_text + c + "\n"
    return prompt_text, full_text
