"""Shared helpers: eval-format prompt construction for gemma-3 GSM8K SFT."""
import os
import re

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "gemma3.jinja")

# Exactly the template used by inspect_evals/gsm8k
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    with open(TEMPLATE_PATH) as f:
        tok.chat_template = f.read()
    return tok


def user_message(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def build_messages(question: str, system: str | None = None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_message(question)})
    return msgs


def eval_fewshot_system(n: int = 10, seed: int = 42) -> str:
    """Reproduce the exact 10-shot system message used by the inspect eval.

    Few-shot examples come from the GSM8K *train* split (never test).
    """
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
    return "\n\n".join([sample_to_fewshot(s) for s in fewshots])


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return s
    if f == int(f):
        return str(int(f))
    return str(f)


def extract_pred(text: str) -> str | None:
    """Mimic inspect's match(location='end', numeric=True): last number in output."""
    nums = NUM_RE.findall(text.replace("**", ""))
    if not nums:
        return None
    return norm_num(nums[-1])
