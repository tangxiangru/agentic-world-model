#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Source: nvidia/OpenMathInstruct-2, rows whose problem_source is gsm8k or
augmented_gsm8k (both are seeded from the GSM8K *train* split; the benchmark
test split is never touched).

Target format is dictated by the grader:
  * inspect_evals/gsm8k wraps the question in MATH_PROMPT_TEMPLATE and scores
    with match(numeric=True, location="end") -> the LAST number in the output
    must be the gold answer.
  * templates/gemma3.jinja terminates an assistant turn with <end_of_turn>.
So every target is  <solution text, \boxed stripped>\n\nANSWER: <n><end_of_turn>
"""
import argparse, json, pickle, random, re, sys

USER_TMPL = (
    'Solve the following math problem step by step. The last line of your response '
    'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    'answer to the problem.\n\n{prompt}\n\nRemember to put your answer on its own line '
    'at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    'answer to the problem, and you do not need to use a \\boxed command.\n\nReasoning:'
)
STOP = "<end_of_turn>"
MARKER = "ANSWER: "

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def strip_boxed(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = BOXED.sub(r"\1", s)
    return s


def render_prompt(question: str, fewshot: str | None) -> str:
    """Reproduce templates/gemma3.jinja for [system?, user] + add_generation_prompt,
    minus the leading bos (the tokenizer adds it)."""
    user = USER_TMPL.format(prompt=question).strip()
    prefix = (fewshot + "\n\n") if fewshot else ""
    return f"<start_of_turn>user\n{prefix}{user}{STOP}\n<start_of_turn>model\n"


def fewshot_block(exemplars) -> str:
    return "\n\n".join(
        f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}" for q, r, a in exemplars
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/omi2_gsm_byprob.pkl")
    ap.add_argument("--out", default="data/sft_pool.jsonl")
    ap.add_argument("--sols-orig", type=int, default=4)
    ap.add_argument("--sols-aug", type=int, default=1)
    # skip the first N solutions of each problem, so a later corpus can be
    # built out of rows the previous one did not contain
    ap.add_argument("--skip-orig", type=int, default=0)
    ap.add_argument("--skip-aug", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- held-out probe questions: never train on them -----------------
    heldout = set()
    for line in open("data/heldout_dev300.jsonl"):
        heldout.add(json.loads(line)["question"].strip())

    # ---- exemplar pool for the fewshot-prefix hedge --------------------
    from datasets import load_dataset
    tr = load_dataset("openai/gsm8k", "main")["train"]
    exemplar_pool = []
    for r in tr:
        if r["question"].strip() in heldout:
            continue
        body, ans = r["answer"].split("####")
        # keep the <<a*b=c>> calculator annotations: inspect_evals'
        # sample_to_fewshot passes record["answer"] through untouched, so the
        # grader's 10-shot prefix has them and this hedge must match it.
        exemplar_pool.append((r["question"].strip(), body.strip(), ans.strip()))
    print(f"exemplar pool {len(exemplar_pool)}", file=sys.stderr)

    byprob = pickle.load(open(args.pkl, "rb"))
    rows, skipped = [], {"heldout": 0, "no_ans": 0, "marker": 0, "tail": 0, "dup": 0}

    for question, sols in byprob.items():
        q = question.strip()
        if q in heldout:
            skipped["heldout"] += 1
            continue
        src = sols[0][2]
        if src == "gsm8k":
            keep, skip = args.sols_orig, args.skip_orig
        else:
            keep, skip = args.sols_aug, args.skip_aug
        seen = set()
        used = 0
        for body, ans, _s in sols[skip:]:
            if used >= keep:
                break
            ans = ans.strip()
            if not ans.replace(".", "").replace("-", "").isnumeric():
                skipped["no_ans"] += 1
                continue
            txt = strip_boxed(body).strip()
            if "ANSWER:" in txt.upper():
                skipped["marker"] += 1
                continue
            key = txt[:200]
            if key in seen:
                skipped["dup"] += 1
                continue
            seen.add(key)
            completion = f"{txt}\n\n{MARKER}{ans}{STOP}"
            # the grader reads the last number: it must be `ans`
            nums = re.findall(r"\d+(?:\.\d+)?", completion.replace(",", ""))
            if not nums or nums[-1] != ans.replace(",", "").lstrip("-"):
                skipped["tail"] += 1
                continue
            rows.append({"question": q, "completion": completion,
                         "answer": ans, "src": src})
            used += 1

    rng.shuffle(rows)
    n_few = int(len(rows) * args.fewshot_frac)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            if i < n_few:
                k = rng.randint(2, 10)
                fs = fewshot_block(rng.sample(exemplar_pool, k))
            else:
                k, fs = 0, None
            f.write(json.dumps({
                "prompt": render_prompt(r["question"], fs),
                "completion": r["completion"],
                "answer": r["answer"], "src": r["src"], "n_shot": k,
            }) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}; skipped {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
