"""Single source of truth for how a training row is rendered.

The grader (evaluate.py -> inspect_evals/gsm8k) renders conversations with
templates/gemma3.jinja, NOT with whatever chat template the tokenizer shipped.
Everything here therefore reads that exact file so training and grading agree
byte-for-byte (pitfall: template_unreachable).
"""
import hashlib
import os

TASK_DIR = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
MODEL_PATH = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

# copied verbatim from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_text():
    with open(TEMPLATE_PATH) as fh:
        return fh.read()


def template_sha():
    return hashlib.sha256(template_text().encode()).hexdigest()[:12]


def get_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.chat_template = template_text()
    return tok


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(tok, question: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_content(question)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render_target(solution: str, answer: str) -> str:
    """Body then the single marker line the grader reads, then the stop token."""
    return f"{solution.strip()}\n{ANSWER_MARKER}{answer}{STOP_TOKEN}"


if __name__ == "__main__":
    tok = get_tokenizer()
    print("template sha256[:12] =", template_sha())
    p = render_prompt(tok, "Janet has 3 apples and buys 4 more. How many?")
    t = render_target("3 + 4 = 7 apples.", "7")
    print("--- PROMPT ---")
    print(repr(p))
    print("--- TARGET ---")
    print(repr(t))
    print("prompt tokens", len(tok(p, add_special_tokens=False).input_ids))
    print(
        "target tokens", len(tok(t, add_special_tokens=False).input_ids)
    )
    ids = tok(t, add_special_tokens=False).input_ids
    print("last target token id", ids[-1], repr(tok.decode([ids[-1]])))
