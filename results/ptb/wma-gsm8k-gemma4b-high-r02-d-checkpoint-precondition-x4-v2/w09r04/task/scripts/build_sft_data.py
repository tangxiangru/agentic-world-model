"""Build the SFT set for GSM8K under the harness's own prompt/answer contract.

Source: nvidia/OpenMathInstruct-2, restricted to problem_source in
{gsm8k, augmented_gsm8k} -- i.e. solutions written for the GSM8K *train*
problems and for problems augmented from them. Nothing here touches the
benchmark test split; ../contamination_check.py is run over the output.

Every row is rendered with templates/gemma3.jinja (the exact file the grader
passes to vLLM), so the training string and the grading string agree
byte-for-byte. The target ends with the harness's answer line and the template's
terminator:  "...\n\nANSWER: 45<end_of_turn>".

A fraction of rows carry a k-shot system message built the way
inspect_evals.gsm8k.sample_to_fewshot builds it, from held-in GSM8K train rows,
so the model sees the eval's long-system-prompt condition during training and
does not copy the demos' terse style.
"""
import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STOP_TOKEN

ROOT = Path(__file__).resolve().parent.parent

# byte-for-byte the harness's prompt (inspect_evals/gsm8k/gsm8k.py L24-38)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUMERIC = re.compile(r"-?\d+(\.\d+)?$")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def norm_num(s: str) -> str:
    return s.strip().replace(",", "").replace("$", "")


def clean_solution(sol: str, answer: str) -> str | None:
    """Turn an OpenMathInstruct-2 solution into a harness-shaped target."""
    if sol.count("\\boxed") != 1:
        return None
    m = BOXED.search(sol)
    if m is None or norm_num(m.group(1)) != answer:
        return None
    body = BOXED.sub(lambda mm: mm.group(1), sol).strip()
    # drop leftover latex the grade-school register does not need
    if "\\begin{" in body or "\\[" in body:
        return None
    body = body.replace("\\times", "*").replace("\\div", "/").replace("\\cdot", "*")
    if "\\" in body:
        return None
    # the grader reads the LAST numeric token: the answer line must be last, and
    # the turn must end with the template's terminator so vLLM stops there
    return f"{body}\n\nANSWER: {answer}{STOP_TOKEN}"


def fewshot_system(pool: list[dict], k: int, rng: random.Random) -> str:
    """Same shape as inspect_evals.gsm8k.sample_to_fewshot, joined with \n\n."""
    shots = rng.sample(pool, k)
    out = []
    for s in shots:
        reasoning = s["answer"].split("####")[0].strip()
        gold = s["answer"].split("####")[-1].strip().replace(",", "")
        out.append(f"{s['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {gold}")
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=40000)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N rows of the seeded shuffle, so a later card can draw solutions the previous one never saw")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "sft_omi2_gsm8k.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    shards = sorted((ROOT.parent / "hf_cache" / "hub").glob(
        "datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-*.parquet"))
    assert shards, "no OpenMathInstruct-2 shards downloaded"
    print("shards:", [s.name for s in shards])

    by_problem: dict[str, list[str]] = {}
    kept = 0
    for path in shards:
        f = pq.ParquetFile(path)
        for rg in range(f.metadata.num_row_groups):
            t = f.read_row_group(rg, columns=[
                "problem", "generated_solution", "expected_answer", "problem_source"
            ]).to_pydict()
            for i in range(len(t["problem"])):
                if t["problem_source"][i] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = norm_num(t["expected_answer"][i])
                if not NUMERIC.fullmatch(ans):
                    continue
                target = clean_solution(t["generated_solution"][i], ans)
                if target is None:
                    continue
                prob = t["problem"][i].strip()
                if len(prob) > 1200 or len(target) > 3000:
                    continue
                sols = by_problem.setdefault(prob, [])
                if len(sols) < args.max_per_problem and target not in sols:
                    sols.append(target)
                    kept += 1
        print(f"  {path.name}: running kept={kept} problems={len(by_problem)}")

    rows = [(p, s) for p, sols in by_problem.items() for s in sols]
    rng.shuffle(rows)
    rows = rows[args.offset: args.offset + args.n_target]
    print(f"kept {kept} solutions over {len(by_problem)} problems -> using {len(rows)} (offset {args.offset})")

    pool = [json.loads(l) for l in (ROOT / "data" / "gsm8k_trainpool.jsonl").open()]

    out = Path(args.out)
    n_fs = 0
    with out.open("w") as fh:
        for prob, target in rows:
            user = MATH_PROMPT_TEMPLATE.format(prompt=prob)
            messages = []
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 3, 4, 5, 8, 10])
                messages.append({"role": "system", "content": fewshot_system(pool, k, rng)})
                n_fs += 1
            messages.append({"role": "user", "content": user})
            fh.write(json.dumps({
                "messages": messages,
                "completion": target,
                "text": prob + "\n" + target,   # what the contamination checker reads
            }) + "\n")
    print(f"wrote {out}  ({n_fs} rows carry a few-shot system message)")


if __name__ == "__main__":
    main()
