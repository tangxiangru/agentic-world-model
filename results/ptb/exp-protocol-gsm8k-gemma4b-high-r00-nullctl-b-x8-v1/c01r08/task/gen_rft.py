#!/usr/bin/env python3
"""Rejection-sampling data generation with vLLM from a fine-tuned checkpoint."""
import argparse, json, os, re, random, collections

PROMPT = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    "{q}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) '
    "where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_num(t):
    ms = NUM.findall(t.replace("$", "").replace("*", ""))
    if not ms:
        return None
    v = ms[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
    except ValueError:
        return None
    return f


def eq(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-4


def norm_sig(sol):
    """Signature for dedup: sequence of numbers appearing in the solution."""
    return tuple(NUM.findall(sol))[-12:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-q", type=int, default=2)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    print("questions:", len(rows))

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()

    prompts = []
    for r in rows:
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(q=r["question"])}],
            tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=2048, dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95,
                        max_tokens=args.max_tokens,
                        stop=["<end_of_turn>", "<eos>"])
    outs = llm.generate(prompts, sp)

    kept, stats = [], collections.Counter()
    solved = 0
    for r, o in zip(rows, outs):
        gold = last_num(str(r["answer"]))
        good = []
        for c in o.outputs:
            txt = c.text.strip()
            if not txt.endswith(tuple("0123456789")):
                # allow trailing period
                txt = txt.rstrip(".")
            if "ANSWER:" not in txt:
                stats["noformat"] += 1
                continue
            if eq(last_num(txt.split("ANSWER:")[-1]), gold):
                good.append(txt)
            else:
                stats["wrong"] += 1
        stats["correct"] += len(good)
        if good:
            solved += 1
        seen, sel = set(), []
        good.sort(key=len)
        for g in good:
            s = norm_sig(g)
            if s in seen:
                continue
            seen.add(s)
            sel.append(g)
            if len(sel) >= args.max_per_q:
                break
        for g in sel:
            kept.append({"question": r["question"], "solution": g,
                         "answer": str(r["answer"])})

    print("stats", dict(stats), "solved_qs", solved, "/", len(rows),
          "kept", len(kept))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    # also dump per-question solve rate for curriculum use
    with open(args.out + ".unsolved", "w") as f:
        for r, o in zip(rows, outs):
            gold = last_num(str(r["answer"]))
            nc = sum(1 for c in o.outputs
                     if "ANSWER:" in c.text and eq(last_num(c.text.split("ANSWER:")[-1]), gold))
            if nc == 0:
                f.write(json.dumps({"question": r["question"], "answer": str(r["answer"])}) + "\n")


if __name__ == "__main__":
    main()
