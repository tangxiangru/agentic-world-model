"""Rejection-sampling fine-tuning data: sample k solutions per problem from the
current checkpoint, keep the ones whose terminal 'ANSWER: N' equals the known
answer, and write them back in the training format.

Problems and gold answers come from the already-built SFT file (GSM8K train and
OpenMathInstruct-2 augmentations of it).  Nothing here reads the benchmark test
split.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")


def question_of(prompt):
    body = prompt.split("<start_of_turn>user\n", 1)[1].rsplit("<end_of_turn>", 1)[0]
    head = "(without quotes) where $ANSWER is the answer to the problem.\n\n"
    q = body.split(head, 1)[-1]
    return q.split("\n\nRemember to put your answer")[0].strip()


def gold_of(completion):
    m = re.search(r"ANSWER:\s*(-?[\d.]+)\s*<end_of_turn>\s*$", completion)
    return m.group(1) if m else None


def norm(x):
    try:
        return format(float(str(x).replace(",", "")), ".5g")
    except (ValueError, TypeError):
        return None


def pred_of(text):
    v = text.strip().replace(",", "").replace("$", "")
    for w in reversed(re.split(r"\s+", v)):
        w = w.strip().rstrip(".").rstrip("%")
        if w.replace(".", "").replace("-", "").isnumeric():
            return norm(w)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True, help="built SFT jsonl to take problems from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=40000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--shots", default="/home/ben/task/data/fewshot10.json")
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-weighted", action="store_true")
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = {}
    with open(args.src) as fh:
        for line in fh:
            r = json.loads(line)
            g = gold_of(r["completion"])
            if g is None:
                continue
            seen.setdefault(question_of(r["prompt"]), g)
    problems = list(seen.items())
    rng.shuffle(problems)
    problems = problems[: args.n_problems]
    print(f"problems: {len(problems)}", flush=True)

    prompts = [fmt.render_prompt(q) for q, _ in problems]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    shots = json.load(open(args.shots))
    n_written = 0
    n_easy = [0]
    n_solved = 0
    n_samples = 0
    n_correct = 0
    with open(args.out, "w") as o:
        for (q, gold), out in zip(problems, outs):
            g = norm(gold)
            kept = []
            for c in out.outputs:
                n_samples += 1
                if c.finish_reason != "stop":
                    continue
                t = c.text.strip()
                if t.count("ANSWER:") != 1 or pred_of(t) != g:
                    continue
                n_correct += 1
                if t not in kept:
                    kept.append(t)
            if not kept:
                continue
            n_solved += 1
            rng.shuffle(kept)
            # keep fewer copies of problems the model already always gets right:
            # the point of the round is to move the hard-but-reachable problems,
            # not to re-teach the easy ones.
            n_keep = args.keep_per_problem
            if args.hard_weighted and len(kept) == args.k:
                n_keep = 1
                n_easy[0] += 1
            for t in kept[:n_keep]:
                system = None
                if rng.random() < args.fewshot_frac:
                    k = rng.choice([2, 4, 10])
                    system = "\n\n".join(rng.sample(shots, min(k, len(shots))))
                comp = fmt.render_completion(t)
                if comp.count("ANSWER:") != 1 or not comp.endswith(fmt.STOP_TOKEN):
                    continue
                o.write(json.dumps({"prompt": fmt.render_prompt(q, system),
                                    "completion": comp,
                                    "n_shot": 0 if system is None else k}) + "\n")
                n_written += 1
    stats = {"problems": len(problems), "samples": n_samples,
             "sample_accuracy": n_correct / max(1, n_samples),
             "problems_solved_at_least_once": n_solved,
             "pass_at_k": n_solved / max(1, len(problems)),
             "rows_written": n_written, "all_k_correct": n_easy[0], "out": args.out,
             "k": args.k, "temperature": args.temperature}
    print(json.dumps(stats, indent=1), flush=True)
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
