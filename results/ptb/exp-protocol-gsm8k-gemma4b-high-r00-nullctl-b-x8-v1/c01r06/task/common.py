"""Shared formatting helpers matching the inspect_evals/gsm8k eval exactly."""
import re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def user_content(question: str, fewshot_prefix: str = "") -> str:
    """Reproduce what gemma3.jinja puts inside the single user turn."""
    body = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    if fewshot_prefix:
        return (fewshot_prefix.strip() + "\n\n" + body).strip()
    return body


def render_prompt(question: str, fewshot_prefix: str = "") -> str:
    """Full string fed to the model, per templates/gemma3.jinja (bos added by tokenizer)."""
    return (
        "<start_of_turn>user\n"
        + user_content(question, fewshot_prefix)
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def fewshot_block(question: str, reasoning: str, target: str) -> str:
    """Same shape as inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def split_gsm8k_answer(ans: str):
    parts = ans.split("####")
    target = parts.pop().strip()
    return "####".join(parts).strip(), target


# ---------------------------------------------------------------- answer utils
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_num(s):
    """Normalise a numeric answer string for comparison (mirrors match(numeric=True))."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")) or abs(v) > 1e15:
        return None
    if v == int(v):
        return str(int(v))
    return str(round(v, 6))


def extract_answer(text: str):
    """Pull the value following the last 'ANSWER:' in a completion."""
    idx = text.rfind("ANSWER:")
    if idx == -1:
        return None
    tail = text[idx + len("ANSWER:"):].strip().split("\n")[0].strip()
    return tail


# ------------------------------------------------------------- latex cleanup
_FRAC_RE = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")


def clean_latex(text: str) -> str:
    """Turn OpenMathInstruct's light LaTeX into plain arithmetic prose."""
    t = text
    for _ in range(3):
        t = _FRAC_RE.sub(r"(\1/\2)", t)
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    t = t.replace("\\%", "%").replace("\\$", "$").replace("\\,", " ")
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    t = t.replace("$", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


_BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Replace \\boxed{X} with X (brace-balanced)."""
    while True:
        m = _BOXED_RE.search(text)
        if not m:
            return text
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        inner = text[m.end(): i - 1]
        text = text[: m.start()] + inner + text[i:]
