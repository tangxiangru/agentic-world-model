#!/usr/bin/env python3
"""Mixture for the second training stage.

Three parts:
  1. rejection-sampled chains from the exp-02 checkpoint (correct answers only)
  2. OpenMathInstruct-2 rows for gsm8k-style problems that exp-02's SFT set
     never saw, so the stage adds coverage and not just self-reinforcement
  3. a replay slice of the exp-02 SFT set, to stop the stage drifting off the
     distribution that already scores 0.7333
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
from transformers import AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", required=True)
    ap.add_argument("--seen", required=True, help="decon view of the stage-1 set (has 'question')")
    ap.add_argument("--replay", required=True, help="stage-1 jsonl to replay from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fresh", type=int, default=20000)
    ap.add_argument("--n-replay", type=int, default=15000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    chat_template = open(TEMPLATE).read()

    rows: list[dict] = []

    # 1. rejection-sampled -----------------------------------------------------
    rft_q = set()
    n_rft = 0
    with open(args.rft) as f:
        for line in f:
            r = json.loads(line)
            rft_q.add(r["question"])
            rows.append({"prompt": r["prompt"], "completion": r["completion"], "src": "rft"})
            n_rft += 1

    # 2. fresh OpenMathInstruct-2 problems ------------------------------------
    seen_q = {json.loads(l)["question"] for l in open(args.seen)}
    by_problem: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in sorted(glob.glob(OMI2)):
        t = pq.read_table(path)
        for r in t.to_pylist():
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            p = r["problem"]
            if p in seen_q or p in rft_q:
                continue
            ans = norm_num(r["expected_answer"])
            if ans is None:
                continue
            box = BOXED.findall(r["generated_solution"])
            if len(box) != 1 or norm_num(box[0]) != ans:
                continue
            body = BOXED.sub(lambda m: m.group(1), r["generated_solution"])
            body = body.replace("\\[", "").replace("\\]", "").strip()
            if not body or "ANSWER:" in body or "####" in body:
                continue
            by_problem[p].append((body, ans))
    fresh_probs = sorted(by_problem)
    rng.shuffle(fresh_probs)
    n_fresh = 0
    for p in fresh_probs:
        if n_fresh >= args.n_fresh:
            break
        sols = by_problem[p]
        uniq, seen_h = [], set()
        for body, ans in sols:
            h = hashlib.md5(body.encode()).hexdigest()
            if h not in seen_h:
                seen_h.add(h)
                uniq.append((body, ans))
        uniq.sort(key=len)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=p)}],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        for body, ans in uniq[: args.max_per_problem]:
            rows.append(
                {"prompt": prompt, "completion": f"{body}\n\nANSWER: {ans}<end_of_turn>", "src": "fresh"}
            )
            n_fresh += 1

    # 3. replay ----------------------------------------------------------------
    replay = [json.loads(l) for l in open(args.replay)]
    rng.shuffle(replay)
    n_replay = 0
    for r in replay[: args.n_replay]:
        rows.append({"prompt": r["prompt"], "completion": r["completion"], "src": "replay"})
        n_replay += 1

    rng.shuffle(rows)
    n_drop = 0
    n_marker = 0
    lens = []
    with open(args.out, "w") as f:
        for r in rows:
            n = len(tok(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"])
            lens.append(n)
            if n > args.max_tokens:
                n_drop += 1
                continue
            assert r["completion"].endswith("<end_of_turn>")
            # a few self-sampled chains quote the instruction's 'ANSWER: $X'
            # inside the reasoning; the grader reads the last number so they
            # would still score, but two markers is the double-format pitfall
            if r["completion"].count("ANSWER: ") != 1:
                n_marker += 1
                continue
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")

    lens.sort()
    print(f"[mix] rft {n_rft} + fresh {n_fresh} + replay {n_replay} = {len(rows)}; "
          f"dropped over {args.max_tokens}: {n_drop} ({n_drop/len(rows):.3%}); "
          f"dropped for a second 'ANSWER: ' marker: {n_marker} ({n_marker/len(rows):.3%})")
    print(f"[mix] token length p50 {lens[len(lens)//2]} p95 {lens[int(len(lens)*.95)]} max {lens[-1]}")
    print(f"[mix] wrote {args.out}")


if __name__ == "__main__":
    main()
