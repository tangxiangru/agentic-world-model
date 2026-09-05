#!/usr/bin/env python3
"""Build exp-04 SFT data: GSM8K-train gold + external MetaMathQA GSM-subset.

MetaMathQA (meta-math/MetaMathQA) augments GSM8K/MATH *train* with extra CoT and
rephrasings. We take ONLY the forward GSM types with clean integer answers
(GSM_AnsAug, GSM_Rephrased); we drop MATH_* (non-integer / domain drift) and
GSM_SV / GSM_FOBAR (self-verification / backward "solve-for-unknown" formats whose
final number is a variable value, not the plain answer the last-number grader wants).

Each row is reformatted to be BYTE-IDENTICAL in shape to the exp-01 gold target:
  prompt     = MATH_PROMPT_TEMPLATE.format(prompt=question)   (same wrapper the grader uses)
  completion = <reasoning> + "\n\nANSWER: <int><end_of_turn>"
so few-shot stop-robustness is preserved (the failure mode that broke exp-02).
"""
import argparse, json, re, random

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

KEEP_TYPES = {"GSM_AnsAug", "GSM_Rephrased"}


def parse_metamath(rows):
    out = []
    for r in rows:
        if r.get("type") not in KEEP_TYPES:
            continue
        resp = r["response"]
        # answer sits after the GSM8K-standard "#### N" marker
        m = re.search(r"####\s*([-\d,\.]+)", resp)
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip().rstrip(".")
        # keep clean integer answers only (matches gold; safest for numeric last-number match)
        if not re.fullmatch(r"-?\d+", ans):
            continue
        # reasoning = everything before the #### marker
        reasoning = resp[:m.start()].strip()
        if len(reasoning) < 10:
            continue
        q = r["query"].strip()
        prompt = MATH_PROMPT_TEMPLATE.format(prompt=q)
        completion = f"{reasoning}\n\nANSWER: {ans}<end_of_turn>"
        out.append({"prompt": prompt, "completion": completion, "target": ans,
                    "src": r.get("type")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metamath", default="data/external/MetaMathQA-395K.json")
    ap.add_argument("--gold", default="data/gsm8k_train_sft.jsonl")
    ap.add_argument("--out", default="data/exp04_gold_metamath.jsonl")
    ap.add_argument("--cap_metamath", type=int, default=35000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mm = json.load(open(args.metamath))
    mm_rows = parse_metamath(mm)
    print(f"[metamath] parsed {len(mm_rows)} usable GSM rows from {len(mm)} total")

    # dedup metamath by (prompt, completion)
    seen = set()
    dedup = []
    for r in mm_rows:
        k = (r["prompt"], r["completion"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    print(f"[metamath] {len(dedup)} after dedup")

    rng = random.Random(args.seed)
    rng.shuffle(dedup)
    mm_sample = dedup[:args.cap_metamath]

    gold = [json.loads(l) for l in open(args.gold)]
    for g in gold:
        g["src"] = "gold"
    print(f"[gold] {len(gold)} rows")

    allrows = gold + mm_sample
    rng.shuffle(allrows)
    with open(args.out, "w") as f:
        for r in allrows:
            f.write(json.dumps({k: r[k] for k in ("prompt", "completion", "target")}) + "\n")
    from collections import Counter
    print(f"[out] wrote {len(allrows)} rows to {args.out} "
          f"(gold={len(gold)}, metamath={len(mm_sample)}, "
          f"gold_share={len(gold)/len(allrows):.3f})")
    print("src mix:", dict(Counter(r["src"] for r in allrows)))
    # sanity: every completion must end with <end_of_turn>
    bad = sum(1 for r in allrows if not r["completion"].rstrip().endswith("<end_of_turn>"))
    print("completions NOT ending in <end_of_turn>:", bad)
    print("--- example metamath completion ---")
    ex = next(r for r in allrows if r["src"] != "gold")
    print(ex["completion"][:500])


if __name__ == "__main__":
    main()
