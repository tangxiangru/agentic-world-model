"""Rejection-sampling round: sample k solutions per TRAIN question from the
current checkpoint, keep the ones whose final number matches gold.

Questions come from the GSM8K train split (minus the held-out dev300) and from
OpenMathInstruct-2 train_1M problems with verified expected_answer.  No
benchmark test item is read.
"""
import argparse, json, os, random, re, sys

NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")
PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
STOP = "<end_of_turn>"


def norm(s):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f == int(f) else round(f, 6)


def last_number(t):
    ms = NUM_RE.findall(t)
    return norm(ms[-1]) if ms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl {id,question,gold}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--keep-per-q", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    tpl = open("/home/ben/task/templates/gemma3.jinja").read()
    items = [json.loads(l) for l in open(a.questions)]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=it["question"])}],
            chat_template=tpl, tokenize=False, add_generation_prompt=True)
        for it in items
    ]
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, dtype="bfloat16",
              max_model_len=2048)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens, seed=0)
    outs = llm.generate(prompts, sp)

    kept, solved, unsolved = 0, 0, []
    seen_bodies = set()
    with open(a.out, "w") as f:
        for it, o in zip(items, outs):
            gold = norm(it["gold"])
            good = []
            for c in o.outputs:
                if c.finish_reason == "length":
                    continue
                txt = c.text.strip()
                if txt.count("ANSWER: ") != 1:
                    continue
                if last_number(txt) != gold:
                    continue
                good.append(txt)
            if not good:
                unsolved.append(it)
                continue
            solved += 1
            good.sort(key=len)                     # prefer the shortest correct chain
            for txt in good[: a.keep_per_q]:
                key = hash(txt[:200])
                if key in seen_bodies:
                    continue
                seen_bodies.add(key)
                f.write(json.dumps({
                    "prompt": PROMPT_TEMPLATE.format(prompt=it["question"]),
                    "target": txt + STOP,
                    "source": "rft_self",
                    "question": it["question"],
                }) + "\n")
                kept += 1
    stats = {"questions": len(items), "solved": solved,
             "solve_rate": solved / len(items), "rows_kept": kept,
             "unsolved": len(unsolved)}
    print(json.dumps(stats, indent=2))
    if a.stats:
        json.dump(stats, open(a.stats, "w"), indent=2)
        with open(a.stats.replace(".json", "_unsolved.jsonl"), "w") as f:
            for it in unsolved:
                f.write(json.dumps(it) + "\n")


if __name__ == "__main__":
    main()
