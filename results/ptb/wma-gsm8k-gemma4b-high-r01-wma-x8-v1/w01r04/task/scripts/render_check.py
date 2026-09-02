#!/usr/bin/env python3
"""Render an eval prompt with the grader's own template and compare it to the
string the trainer will build. Catches the template_unreachable pitfall."""
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def grader_template() -> str:
    return open(TEMPLATE).read()


def fewshot_system() -> str:
    """Rebuild the exact 10-shot system message inspect_evals/gsm8k builds."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42).select(range(10))
    parts = []
    for r in ds:
        q = r["question"]
        answer = r["answer"].split("####")
        target = answer.pop().strip()
        reasoning = "####".join(answer).strip()
        parts.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return "\n\n".join(parts)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(SNAP)
    tpl = grader_template()
    print("template sha256:", hashlib.sha256(tpl.encode()).hexdigest()[:16])

    q = "Janet has 3 boxes with 12 apples each. She gives away 7 apples. How many are left?"
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)

    # --- what the grader sends (zero-shot form, for string comparison) ---
    zs = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        chat_template=tpl,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("\n=== grader render (zero-shot) ===")
    print(repr(zs))

    # --- what the trainer will build ---
    trainer_prompt = (
        f"{tok.bos_token}<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
    )
    print("\ntrainer prompt matches grader render:", trainer_prompt == zs)
    if trainer_prompt != zs:
        print("trainer:", repr(trainer_prompt))
        sys.exit(1)

    # --- the real thing the grader sends: 10-shot system prefix ---
    sysmsg = fewshot_system()
    fs = tok.apply_chat_template(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
        chat_template=tpl,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(fs, add_special_tokens=False)["input_ids"]
    print("\n=== grader render (10-shot) ===")
    print("prompt tokens:", len(ids))
    print("head:", repr(fs[:300]))
    print("tail:", repr(fs[-700:]))
    print("bos duplicated:", ids.count(tok.bos_token_id))

    # --- terminator / eos facts ---
    gen = json.load(open(os.path.join(SNAP, "generation_config.json")))
    print("\ngeneration_config:", gen)
    for t in ["<end_of_turn>", "<eos>", "<bos>", "<start_of_turn>"]:
        print(f"  {t} -> {tok.convert_tokens_to_ids(t)}")

    completion = "3 * 12 = 36 apples.\n36 - 7 = 29 apples.\n\nANSWER: 29<end_of_turn>\n"
    cid = tok(completion, add_special_tokens=False)["input_ids"]
    print("\ncompletion tail ids:", cid[-6:], tok.convert_ids_to_tokens(cid[-6:]))

    os.makedirs("/home/ben/task/analysis", exist_ok=True)
    with open("/home/ben/task/analysis/render_check.json", "w") as f:
        json.dump(
            {
                "template_sha256": hashlib.sha256(tpl.encode()).hexdigest(),
                "trainer_matches_grader": trainer_prompt == zs,
                "fewshot_prompt_tokens": len(ids),
                "fewshot_system_chars": len(sysmsg),
                "eos_token_ids": gen["eos_token_id"],
                "end_of_turn_id": tok.convert_tokens_to_ids("<end_of_turn>"),
            },
            f,
            indent=2,
        )
    print("\nwrote /home/ben/task/analysis/render_check.json")


if __name__ == "__main__":
    main()
