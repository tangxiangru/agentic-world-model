#!/usr/bin/env python3
"""Filter raw sampled generations into verified RFT training data."""
import argparse
import json
import random
import re
from collections import Counter

from prep_sft import render_prompt, norm_answer

ANS_LINE = re.compile(r"^ANSWER:\s*(.+?)\s*$", re.M)


def truncate_at_answer(text: str):
    """Return (clean_text, answer) using the FIRST well-formed ANSWER line."""
    m = ANS_LINE.search(text)
    if not m:
        return None, None
    a = norm_answer(m.group(1))
    if a is None:
        return None, None
    return text[: m.end()].strip(), a


def sig(sol: str) -> str:
    return ",".join(re.findall(r"-?\d+\.?\d*", sol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fr = Counter()
    n_items = n_solved = kept = 0
    rates = []
    out_f = open(args.out, "w")
    for path in args.raw:
        for line in open(path):
            r = json.loads(line)
            n_items += 1
            good, n_ok = [], 0
            for g in r["gens"]:
                fr[g["fr"]] += 1
                clean, a = truncate_at_answer(g["text"])
                if a != r["gold"]:
                    continue
                n_ok += 1
                body = clean[: clean.rfind("ANSWER:")].strip()
                if len(body) < 40:
                    continue
                good.append(clean)
            rate = n_ok / max(1, len(r["gens"]))
            rates.append(rate)
            if n_ok:
                n_solved += 1
            budget = args.keep if rate < 0.9 else max(1, args.keep // 2)
            rng.shuffle(good)
            seen, chosen = set(), []
            for g in good:
                s = sig(g)
                if s in seen:
                    continue
                seen.add(s)
                chosen.append(g)
                if len(chosen) >= budget:
                    break
            for g in chosen:
                out_f.write(json.dumps({
                    "prompt_text": render_prompt(r["question"], r["system"]),
                    "completion_text": g + "<end_of_turn>",
                    "question": r["question"],
                    "answer": r["gold"],
                }) + "\n")
                kept += 1
    out_f.close()
    print(f"items={n_items} solved={n_solved} kept={kept} "
          f"mean_pass={sum(rates)/len(rates):.3f} finish_reasons={dict(fr)}")


if __name__ == "__main__":
    main()
