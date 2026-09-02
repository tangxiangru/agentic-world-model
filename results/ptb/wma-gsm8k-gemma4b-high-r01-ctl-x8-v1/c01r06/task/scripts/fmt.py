"""Shared formatting: renders exactly what the grader renders.

The grader (evaluate.py -> inspect_evals/gsm8k -> vllm server) builds a chat
conversation and renders it with templates/gemma3.jinja.  Everything here is
derived from those two files so training and grading cannot drift apart
(pitfall: template_unreachable).
"""
import hashlib
import os
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# byte-for-byte hash of the template the grader uses; asserted at import time by
# callers that care (build_data.py, train_sft.py).
with open(TEMPLATE_PATH, "rb") as _f:
    TEMPLATE_BYTES = _f.read()
TEMPLATE_SHA256 = hashlib.sha256(TEMPLATE_BYTES).hexdigest()

# copied verbatim from inspect_evals/gsm8k/gsm8k.py::MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """Exactly inspect_evals/gsm8k/gsm8k.py::sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def render_prompt(question: str, system: str | None = None) -> str:
    """Render the grader's prompt string for one question.

    Mirrors templates/gemma3.jinja for the (optional system) + single user turn
    case, with add_generation_prompt=True.  vLLM's chat endpoint tokenizes the
    rendered string with add_special_tokens=False, so the single <bos> comes
    from the template itself.
    """
    prefix = (system.strip() + "\n\n") if system else ""
    return (
        "<bos><start_of_turn>user\n"
        + prefix
        + user_content(question).strip()
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def render_completion(solution: str) -> str:
    return solution.strip() + STOP_TOKEN


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_CALC = re.compile(r"<<[^>]*>>")
_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def clean_number(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not _NUM.match(s):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_solution(sol: str) -> str:
    """Strip artefacts that would give the grader a second answer marker."""
    sol = _BOXED.sub(r"\1", sol)          # \boxed{45} -> 45
    sol = _CALC.sub("", sol)              # <<16-3-4=9>> -> ''
    sol = sol.replace("$", "")            # bare LaTeX/currency markers
    sol = re.sub(r"\n?####.*", "", sol)   # gsm8k's own '#### N' line
    sol = re.sub(r"[ \t]+\n", "\n", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def build_target(solution: str, answer: str) -> str | None:
    """Body + exactly one 'ANSWER: N' terminal line, or None if unusable."""
    ans = clean_number(answer)
    if ans is None:
        return None
    body = normalize_solution(solution)
    if not body:
        return None
    if "ANSWER:" in body:
        return None
    return f"{body}\n\nANSWER: {ans}"
