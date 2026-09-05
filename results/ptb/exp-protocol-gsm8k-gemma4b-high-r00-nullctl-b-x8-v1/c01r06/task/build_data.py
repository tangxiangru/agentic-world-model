"""Build the SFT corpus: math word problems -> eval-format CoT ending in 'ANSWER: N'."""
import json
import random
import re
import sys

from datasets import load_dataset

from common import clean_latex, norm_num, split_gsm8k_answer, strip_boxed

random.seed(0)

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/sft_v1.jsonl"

# Cap per source (None = take all available)
N_AUG_GSM = 140_000
N_GSM = 20_000
N_MATH = 30_000


def finalise(solution: str, answer: str):
    """Clean a generated_solution and make it end with a bare 'ANSWER: x' line."""
    sol = strip_boxed(solution)
    sol = clean_latex(sol)
    # Drop a trailing 'The answer is ...' style sentence -- we append our own line.
    lines = [l.rstrip() for l in sol.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    sol = "\n".join(lines).strip()
    if not sol:
        return None
    return sol + "\n\nANSWER: " + answer


def take(ds, sources, cap, records, seen):
    pool = [i for i, s in enumerate(ds["problem_source"]) if s in sources]
    random.shuffle(pool)
    kept = 0
    probs = ds["problem"]
    sols = ds["generated_solution"]
    answers = ds["expected_answer"]
    for i in pool:
        if kept >= cap:
            break
        ans = norm_num(answers[i])
        if ans is None:
            continue                      # keep the ANSWER: field strictly numeric
        q = probs[i].strip()
        if len(q) < 20 or "[asy]" in q or "\\begin" in q:
            continue
        body = finalise(sols[i], ans)
        if body is None or len(body) > 4000:
            continue
        if "boxed" in body or "\\" in body:
            continue                      # leftover latex -> skip, plenty of data
        key = (q, body)
        if key in seen:
            continue
        seen.add(key)
        records.append({"question": q, "solution": body, "answer": ans,
                        "source": ds["problem_source"][i]})
        kept += 1
    print(f"  {sources}: kept {kept} / pool {len(pool)}", file=sys.stderr)


def main():
    records = []
    seen = set()

    print("loading OpenMathInstruct-2 train_1M ...", file=sys.stderr)
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    take(omi, {"augmented_gsm8k"}, N_AUG_GSM, records, seen)
    take(omi, {"gsm8k"}, N_GSM, records, seen)
    take(omi, {"math", "augmented_math"}, N_MATH, records, seen)

    # Original GSM8K train rationales (terse, matches the eval's few-shot style).
    print("loading gsm8k train ...", file=sys.stderr)
    g = load_dataset("openai/gsm8k", "main", split="train")
    n = 0
    for q, a in zip(g["question"], g["answer"]):
        reasoning, target = split_gsm8k_answer(a)
        reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
        tgt = norm_num(target)
        if tgt is None:
            continue
        records.append({"question": q.strip(),
                        "solution": reasoning + "\n\nANSWER: " + tgt,
                        "answer": tgt, "source": "gsm8k_orig"})
        n += 1
    print(f"  gsm8k_orig: kept {n}", file=sys.stderr)

    random.shuffle(records)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
