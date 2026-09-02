#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per training problem from a checkpoint,
keep the ones whose final 'ANSWER: n' matches the gold answer.

Prompts are rendered with the grader's own template so the samples are drawn
from the same conditional the eval will use. Targets are written in exactly the
schema build_sft_data.py emits, so train_sft.py can read either file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict

TASK_DIR = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates/gemma3.jinja")
END_OF_TURN = "<end_of_turn>"
ANSWER_RE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)\s*$")

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def gold_of(target: str) -> str | None:
    body = target.replace(END_OF_TURN, "").rstrip()
    m = ANSWER_RE.search(body)
    return m.group(1).replace(",", "") if m else None


FIRST_ANSWER_RE = re.compile(r"^ANSWER:[ \t]*(-?[\d,]+(?:\.\d+)?)[ \t]*$", re.M)


def truncate_at_answer(text: str) -> str:
    """Cut the completion at the end of its first well-formed answer line.

    A sample that reached the right answer and then kept going is a good sample
    with a tail; the tail is exactly what the stop token would have removed.
    """
    m = FIRST_ANSWER_RE.search(text)
    return text[: m.end()].strip() if m else text.strip()


def filter_raw(raw_path: str, max_per_problem: int):
    """Keep the samples whose final 'ANSWER: n' equals gold. Counts every drop
    reason so a low keep rate can be attributed instead of guessed at."""
    from collections import Counter

    reasons = Counter()
    kept: dict[str, list[str]] = defaultdict(list)
    n_gen = n_ok = solved = 0
    for line in open(raw_path):
        r = json.loads(line)
        gold = r["gold"]
        good = []
        for s in r["samples"]:
            n_gen += 1
            text = truncate_at_answer(s["text"])
            m = ANSWER_RE.search(text)
            if not m:
                reasons["no_answer_line"] += 1
                continue
            if m.group(1).replace(",", "") != gold:
                reasons["wrong_answer"] += 1
                continue
            if len(text) < 40:
                reasons["too_short"] += 1
                continue
            if text.count("ANSWER:") != 1:
                reasons["multiple_answer_markers"] += 1
                continue
            if s["finish_reason"] != "stop":
                # the text still ends at the answer, so it is usable; count it
                reasons["kept_despite_finish_reason"] += 1
            good.append(text + END_OF_TURN)
            n_ok += 1
        if good:
            solved += 1
        kept[r["question"]] = sorted(set(good), key=len)[:max_per_problem]
    return kept, reasons, n_gen, n_ok, solved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="/home/ben/task/data/sft_v2.jsonl",
                    help="jsonl of {question, target}; target supplies the gold answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=25000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--probe", type=int, default=300)
    ap.add_argument("--min-probe-rate", type=float, default=0.35)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(TEMPLATE_PATH).read()

    rows = [json.loads(l) for l in open(args.pool)]
    # one entry per unique problem
    by_q = {}
    for r in rows:
        g = gold_of(r["target"])
        if g is not None:
            by_q.setdefault(r["question"], g)
    items = sorted(by_q.items())
    random.Random(args.seed).shuffle(items)
    items = items[: args.n_problems]
    print(f"sampling {args.k} completions for {len(items)} problems")

    # templates/gemma3.jinja already emits <bos>. Handing vllm the *string* makes
    # it tokenize with add_special_tokens=True and prepend a second <bos>
    # ([2, 2, 105, ...]), which collapses gemma's accuracy: the first attempt at
    # this sampling run solved 3.2% of problems that the same checkpoint solves
    # 58% of through the grader. Pass token ids instead.
    prompts = []
    for q, _ in items:
        text = tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert ids[:2] != [tok.bos_token_id, tok.bos_token_id], "double bos"
        prompts.append({"prompt_token_ids": ids})

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1024, enable_prefix_caching=True, seed=args.seed,
              # vllm 0.11 sizes the sampled-token pinned buffer by max_model_len,
              # so max_num_seqs above it dies with a tensor-size mismatch in
              # gpu_model_runner._to_list.
              max_num_seqs=768, max_num_batched_tokens=32768)
    # no per-request seed: it forces vllm onto a per-sequence torch Generator and
    # roughly halves sampling throughput. LLM(seed=) already makes the run
    # reproducible.
    # <end_of_turn> (106) is in generation_config.eos_token_id, which the grader's
    # vllm *server* honours but the offline LLM class did not: samples ran on past
    # the answer repeating "ANSWER: n" until max_tokens. Name the stop ids here.
    eot = tok.convert_tokens_to_ids(END_OF_TURN)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        stop_token_ids=[eot, tok.eos_token_id])
    # probe first: a cheap early read on the solve rate, so a broken prompt
    # cannot burn the whole sampling budget again
    n_probe = min(args.probe, len(prompts))
    outs = list(llm.generate(prompts[:n_probe], sp))
    hit = sum(
        any(ANSWER_RE.search(c.text.strip()) and
            ANSWER_RE.search(c.text.strip()).group(1).replace(",", "") == gold
            for c in o.outputs)
        for (q, gold), o in zip(items[:n_probe], outs)
    )
    print(f"PROBE: {hit}/{n_probe} problems solved by at least one of k={args.k} "
          f"({hit/max(n_probe,1):.1%})", flush=True)
    if hit / max(n_probe, 1) < args.min_probe_rate:
        raise SystemExit(
            f"probe solve rate {hit/max(n_probe,1):.1%} below --min-probe-rate "
            f"{args.min_probe_rate}; the prompt or checkpoint is wrong, not the data"
        )
    outs += list(llm.generate(prompts[n_probe:], sp))

    # dump every generation before filtering: re-deriving the corpus offline costs
    # seconds, re-generating it costs 45 minutes of H100 time.
    raw_path = args.out + ".raw.jsonl"
    with open(raw_path, "w") as rf:
        for (q, gold), o in zip(items, outs):
            rf.write(json.dumps({
                "question": q, "gold": gold,
                "samples": [{"text": c.text, "finish_reason": c.finish_reason}
                            for c in o.outputs],
            }) + "\n")
    print("raw generations ->", raw_path, flush=True)

    kept, reasons, n_gen, n_ok, solved = filter_raw(raw_path, args.max_per_problem)
    print("filter drops:", dict(reasons), flush=True)

    with open(args.out, "w") as f:
        n = 0
        for q, tgts in kept.items():
            for t in tgts:
                f.write(json.dumps({"question": q, "target": t, "source": "rft"}) + "\n")
                n += 1
    print(f"generated={n_gen} correct={n_ok} ({n_ok/max(n_gen,1):.1%}) "
          f"problems_with_a_correct_sample={solved}/{len(items)} "
          f"({solved/len(items):.1%}) rows_written={n} -> {args.out}")

    stats = {
        "model": args.model, "n_problems": len(items), "k": args.k,
        "temperature": args.temperature, "generated": n_gen, "correct": n_ok,
        "pass_at_k_problems": solved, "rows_written": n,
        "template_sha256_12": hashlib.sha256(open(TEMPLATE_PATH, "rb").read()).hexdigest()[:12],
    }
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
