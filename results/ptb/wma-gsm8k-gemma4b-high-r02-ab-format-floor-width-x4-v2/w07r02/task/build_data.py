#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Sources
-------
* nvidia/OpenMathInstruct-2 (train_1M split), rows whose problem_source is
  gsm8k / augmented_gsm8k (and optionally a slice of augmented_math).
  Those problems are derived from the GSM8K/MATH *train* splits only.
* openai/gsm8k main/train  -- used only as the pool for the few-shot prefixes
  that a fraction of the training prompts carry, so the model sees the same
  prompt shape the grader uses.

Output rows carry four keys:
  prompt      -- the fully rendered chat string, byte-identical to what the
                 grader's templates/gemma3.jinja produces (add_generation_prompt)
  completion  -- solution text + "<end_of_turn>"
  question    -- raw question   (read by ../contamination_check.py)
  answer      -- raw solution   (read by ../contamination_check.py)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq
from jinja2 import Environment
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

# Exactly the wording inspect_evals/gsm8k puts in the user turn.
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
EOT = "<end_of_turn>"


def load_template():
    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
    env.policies["json.dumps_kwargs"] = {}

    def raise_exception(msg):
        raise ValueError(msg)

    src = open(TEMPLATE_PATH).read()
    tpl = env.from_string(src)
    tpl.globals["raise_exception"] = raise_exception
    return tpl


TPL = load_template()


def render_prompt(system: str | None, user: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return TPL.render(messages=msgs, bos_token=BOS, add_generation_prompt=True)


# --------------------------------------------------------------------------
# solution cleaning
# --------------------------------------------------------------------------
BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-matched)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        out.append(text[m.end(): j - 1])
        i = j
    return "".join(out)


TRAILING_MATH_JUNK = re.compile(r"[\s\$\\\[\]\(\)\{\}=\.,:;]+$")


def clean_solution(sol: str, answer: str) -> str | None:
    """Turn an OpenMathInstruct-2 solution into a target that ends in
    'ANSWER: <answer>' exactly once, with no number after it."""
    sol = strip_boxed(sol).strip()
    # Drop a trailing bare restatement of the answer, e.g. "\\[ y = 2 \\]" or
    # "The answer is 2." -- we re-add a canonical one.
    lines = sol.split("\n")
    while lines:
        last = lines[-1].strip()
        stripped = TRAILING_MATH_JUNK.sub("", last)
        # a final line that is only the answer (possibly with 'x =' / prose)
        compact = re.sub(r"[\s\$\\\[\]]", "", last)
        if not last:
            lines.pop()
            continue
        if compact and compact.rstrip(".").endswith(answer.replace(" ", "")) and len(compact) <= len(answer) + 24:
            lines.pop()
            continue
        break
    sol = "\n".join(lines).strip()
    if not sol:
        return None
    if "ANSWER:" in sol:
        return None
    sol = sol + f"\n\nANSWER: {answer}"
    return sol


NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def numeric_ok(ans: str) -> bool:
    return bool(NUMERIC_RE.match(ans.strip().replace(",", "")))


def last_number(text: str) -> str | None:
    """Mimic inspect's match(location='end', numeric=True) extraction."""
    v = text.replace(",", "").replace("$", "").replace("%", "")
    words = re.split(r"\s+", v.strip())
    for w in reversed(words):
        w2 = w.strip(".!?):;'\"")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return w2
    return None


# --------------------------------------------------------------------------
def build_fewshot_pool():
    path = sorted(glob.glob(GSM8K_TRAIN))[0]
    tbl = pq.read_table(path).to_pylist()
    pool = []
    for r in tbl:
        q = r["question"].strip()
        a = r["answer"]
        if "####" not in a:
            continue
        cot, ans = a.rsplit("####", 1)
        # keep the <<a*b=c>> calculator annotations: inspect_evals renders the
        # few-shot rationales straight from record["answer"], braces and all.
        cot = cot.strip()
        pool.append((q, cot, ans.strip()))
    return pool


def fewshot_block(pool, rng, k):
    picks = rng.sample(pool, k)
    return "\n\n".join(
        f"{q}\n\nReasoning:\n{cot}\n\nANSWER: {a}" for q, cot, a in picks
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=60000)
    ap.add_argument("--n-math", type=int, default=12000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.20)
    ap.add_argument("--fewshot-max-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = build_fewshot_pool()
    print(f"few-shot pool: {len(pool)} gsm8k train items")

    gsm_rows, math_rows = [], []
    per_problem = defaultdict(int)
    seen_sol = set()
    src_counts = defaultdict(int)

    files = sorted(glob.glob(OMI_GLOB))
    assert files, "OpenMathInstruct-2 shards not found"
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                src = r["problem_source"]
                src_counts[src] += 1
                is_gsm = src in ("gsm8k", "augmented_gsm8k")
                is_math = src in ("math", "augmented_math")
                if not (is_gsm or is_math):
                    continue
                if is_gsm and len(gsm_rows) >= args.n_gsm:
                    if len(math_rows) >= args.n_math:
                        continue
                if is_math and len(math_rows) >= args.n_math:
                    continue
                ans = r["expected_answer"]
                if not numeric_ok(ans):
                    continue
                ans = ans.strip().replace(",", "")
                key = hashlib.md5(r["problem"].strip().encode()).hexdigest()
                if per_problem[key] >= args.max_per_problem:
                    continue
                tgt = clean_solution(r["generated_solution"], ans)
                if tgt is None:
                    continue
                if last_number(tgt) != last_number("x " + ans):
                    continue
                if tgt.count("ANSWER:") != 1:
                    continue
                h = hashlib.md5(tgt.encode()).hexdigest()
                if h in seen_sol:
                    continue
                seen_sol.add(h)
                per_problem[key] += 1
                rec = {"question": r["problem"].strip(), "answer": tgt, "src": src}
                if is_gsm and len(gsm_rows) < args.n_gsm:
                    gsm_rows.append(rec)
                elif is_math and len(math_rows) < args.n_math:
                    math_rows.append(rec)
            if len(gsm_rows) >= args.n_gsm and len(math_rows) >= args.n_math:
                break
        if len(gsm_rows) >= args.n_gsm and len(math_rows) >= args.n_math:
            break

    print("source counts seen:", dict(src_counts))
    print(f"kept gsm={len(gsm_rows)} math={len(math_rows)}")

    rows = gsm_rows + math_rows
    rng.shuffle(rows)

    n_fs = 0
    with open(args.out, "w") as fh:
        for rec in rows:
            user = MATH_PROMPT_TEMPLATE.format(prompt=rec["question"])
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, args.fewshot_max_k)
                system = fewshot_block(pool, rng, k)
                n_fs += 1
            prompt = render_prompt(system, user)
            fh.write(json.dumps({
                "prompt": prompt,
                "completion": rec["answer"] + EOT,
                "question": rec["question"],
                "answer": rec["answer"],
                "src": rec["src"],
                "fewshot": system is not None,
            }) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}; {n_fs} carry a few-shot prefix")


if __name__ == "__main__":
    main()
