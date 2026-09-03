#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training question
from a checkpoint, keep the ones whose ANSWER matches gold, dedup by reasoning
path, write an SFT jsonl with the same schema as data/sft_v2.jsonl.

Prompts are rendered with the grader's own templates/gemma3.jinja and the
inspect_evals MATH_PROMPT_TEMPLATE, so the samples are on-policy for the eval.
"""
import argparse, json, os, random, re, sys
from collections import defaultdict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sft import MATH_PROMPT_TEMPLATE, STOP  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(x):
    try:
        f = float(str(x).replace(",", "").rstrip("."))
    except ValueError:
        return None
    return round(f, 4)


def final_answer(text):
    """Same rule the grader uses: the last number in the completion."""
    m = NUM.findall(text)
    return norm(m[-1]) if m else None


def path_key(sol):
    """Dedup key: the sequence of numbers that appear in the reasoning."""
    return tuple(norm(x) for x in NUM.findall(sol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with q/gold fields")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep-per-question", type=int, default=4)
    ap.add_argument("--template", default="templates/gemma3.jinja")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} questions x k={args.k}", flush=True)

    template = open(args.template).read()
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for r in rows:
        msgs = [{"role": "user",
                 "content": MATH_PROMPT_TEMPLATE.replace("{prompt}", r["q"])}]
        prompts.append(tok.apply_chat_template(msgs, chat_template=template,
                                               tokenize=False, add_generation_prompt=True))
    print("=== example prompt tail ===\n" + repr(prompts[0][-300:]), flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, dtype="bfloat16", seed=args.seed,
              enable_prefix_caching=False, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=16384)
    # vllm's OFFLINE engine stops only on tokenizer.eos_token_id (<eos>, id 1);
    # the served path adds the chat template's terminator itself. Without this the
    # model emits 'ANSWER: n' and keeps going, and 58% of samples were discarded.
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=None,
                        stop_token_ids=[eot])
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept, n_correct, n_total, solved = [], 0, 0, 0
    for r, o in zip(rows, outs):
        gold = norm(r["gold"])
        cands = {}
        for c in o.outputs:
            n_total += 1
            t = c.text.strip()
            if not t.endswith(STOP) and c.finish_reason != "stop":
                continue
            t = t.replace(STOP, "").strip()
            if final_answer(t) != gold:
                continue
            if t.count("ANSWER: ") != 1 or not re.search(r"ANSWER: -?[\d,.]+\s*$", t):
                continue
            n_correct += 1
            cands.setdefault(path_key(t), t)
        if not cands:
            continue
        solved += 1
        picks = list(cands.values())
        rng.shuffle(picks)
        for t in picks[:args.keep_per_question]:
            kept.append({"question": r["q"], "solution": t.rsplit("\n\nANSWER:", 1)[0].strip(),
                         "answer": r["gold"], "src": "rft",
                         "completion": t + STOP})

    with open(args.out, "w") as fh:
        for k in kept:
            fh.write(json.dumps(k) + "\n")
    stats = {"questions": len(rows), "k": args.k, "samples": n_total,
             "correct_samples": n_correct, "pass_rate": n_correct / max(1, n_total),
             "questions_solved": solved, "solve_rate": solved / max(1, len(rows)),
             "kept_rows": len(kept)}
    print(json.dumps(stats, indent=2), flush=True)
    json.dump(stats, open(args.out + ".stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
