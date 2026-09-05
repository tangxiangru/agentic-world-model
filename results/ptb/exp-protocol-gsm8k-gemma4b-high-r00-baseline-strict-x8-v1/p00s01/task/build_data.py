#!/usr/bin/env python3
"""Build GSM8K-style SFT data in the exact format the inspect_evals grader uses.

Source: nvidia/OpenMathInstruct-2 (rows whose problem_source is gsm8k or
augmented_gsm8k -- both are derived from the GSM8K *train* split only).
Few-shot demonstration pool: openai/gsm8k *train* split.
Nothing from the GSM8K test split is read here.

Output jsonl fields:
  prompt      -- rendered exactly as templates/gemma3.jinja renders it at eval time
  completion  -- chain of thought, final line "ANSWER: <n>", terminated by <end_of_turn>
  question    -- raw problem text   (for ../contamination_check.py)
  answer      -- raw solution text  (for ../contamination_check.py)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset

# --- byte-for-byte copies of the strings the grader builds -------------------
# inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP = "<end_of_turn>"


def sample_to_fewshot(question: str, reasoning: str, target: str) -> str:
    """inspect_evals/gsm8k/gsm8k.py sample_to_fewshot"""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def render(system: str | None, user: str) -> str:
    """templates/gemma3.jinja, add_generation_prompt=True, single user turn."""
    first_user_prefix = (system + "\n\n") if system else ""
    return (
        "<bos><start_of_turn>user\n"
        + first_user_prefix
        + user.strip()
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


# --- solution cleanup --------------------------------------------------------
BOXED = re.compile(r"\\boxed\s*{")


def unbox(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-matched)."""
    while True:
        m = BOXED.search(text)
        if not m:
            return text
        i = m.end()  # just after '{'
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth:  # unbalanced; give up
            return text.replace("\\boxed", "")
        text = text[: m.start()] + text[m.end() : i - 1] + text[i:]


NUM_RE = re.compile(r"^-?\d{1,12}(\.\d{1,6})?$")


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("\\!", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    if "." in a:
        a = a.rstrip("0").rstrip(".")
    return a or None


def build_completion(solution: str, answer: str) -> str | None:
    s = unbox(solution).strip()
    if "ANSWER:" in s or "answer:" in s.lower() and "ANSWER:" in s:
        return None
    if "ANSWER:" in s:
        return None
    # drop a trailing bare restatement of the answer so the body reads cleanly
    s = re.sub(r"\n*(So,? |Therefore,? )?[Tt]he answer is[: ]*[^\n]*$", "", s).strip()
    s = re.sub(r"\n*\\\[\s*" + re.escape(answer) + r"\s*\\\]\s*$", "", s).strip()
    if not s:
        return None
    return f"{s}\n\nANSWER: {answer}{STOP}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # few-shot demo pool: GSM8K TRAIN only
    train = load_dataset("openai/gsm8k", "main", split="train")
    demos = []
    for r in train:
        parts = r["answer"].split("####")
        demos.append((r["question"], "####".join(parts[:-1]).strip(), parts[-1].strip()))

    from huggingface_hub import hf_hub_download

    rows: dict[str, list[tuple[str, str]]] = {}
    kept = 0
    for i in range(args.shards):
        p = hf_hub_download(
            "nvidia/OpenMathInstruct-2",
            f"data/train_1M-0000{i}-of-00003.parquet",
            repo_type="dataset",
            revision="469216e3f46f4dacf476b382e192485ea51a143e",
        )
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(
            batch_size=20000,
            columns=["problem", "generated_solution", "expected_answer", "problem_source"],
        ):
            d = batch.to_pydict()
            for prob, sol, ans, src in zip(
                d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
            ):
                if src not in ("gsm8k", "augmented_gsm8k"):
                    continue
                a = clean_answer(ans)
                if a is None:
                    continue
                bucket = rows.setdefault(prob, [])
                if len(bucket) >= args.max_per_problem:
                    continue
                c = build_completion(sol, a)
                if c is None:
                    continue
                bucket.append((c, a))
                kept += 1
        print(f"shard {i}: {len(rows)} problems, {kept} solutions", flush=True)

    flat = [(prob, c, a) for prob, sols in rows.items() for (c, a) in sols]
    rng.shuffle(flat)
    flat = flat[: args.n]
    print(f"writing {len(flat)} rows from {len(rows)} unique problems")

    with open(args.out, "w") as f:
        for prob, comp, a in flat:
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(2, 10)
                picks = rng.sample(demos, k)
                system = "\n\n".join(sample_to_fewshot(*p) for p in picks)
            prompt = render(system, MATH_PROMPT_TEMPLATE.format(prompt=prob))
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": comp,
                        "question": prob,
                        "answer": comp[: -len(STOP)],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
