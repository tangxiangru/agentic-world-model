"""Turn gen.py rejection samples into an SFT file.

Keeps only samples whose final answer matches gold, dedups near-identical
reasoning per question, caps how many are kept per question, and writes the
same {prompt, completion} schema train_sft.py reads (target already carries the
'ANSWER: N' line and the <end_of_turn> terminator).
"""
import argparse
import json
import random
import re

import harness_format as hf


def sig(text):
    """Signature for near-duplicate detection: the sequence of numbers used."""
    return tuple(re.findall(r"-?\d+\.?\d*", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="gen.py output jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    kept, n_q, n_q_any, n_cand = 0, 0, 0, 0
    with open(args.out, "w") as out:
        for line in open(args.samples):
            r = json.loads(line)
            n_q += 1
            good = []
            for c, ok in zip(r["completions"], r["correct"]):
                n_cand += 1
                if not ok:
                    continue
                c = c.strip()
                if len(c) > args.max_chars or "ANSWER:" not in c:
                    continue
                # the answer line must be the last line and appear once
                if c.count("ANSWER:") != 1 or not re.match(
                    r"^ANSWER:\s*-?[\d,]+(\.\d+)?$", c.split("\n")[-1].strip()
                ):
                    continue
                good.append(c)
            if not good:
                continue
            n_q_any += 1
            # difficulty balancing: a problem the model always gets right teaches
            # less than one it only sometimes gets right, so keep fewer of the easy ones
            cap = 1 if len(good) == len(r["completions"]) else args.max_per_question
            rng.shuffle(good)
            seen = set()
            picked = []
            for c in good:
                s = sig(c)
                if s in seen:
                    continue
                seen.add(s)
                picked.append(c)
                if len(picked) >= cap:
                    break
            for c in picked:
                prompt = hf.PROMPT_TEMPLATE.format(prompt=r["question"].strip())
                out.write(json.dumps({
                    "prompt": prompt,
                    "completion": c + hf.STOP_TOKEN,
                    "question": r["question"],
                    "answer": r["gold"],
                    "source": "rft_self",
                }) + "\n")
                kept += 1
    print(f"questions={n_q} solved_at_least_once={n_q_any} ({n_q_any / n_q:.3f}) "
          f"candidates={n_cand} kept={kept} -> {args.out}")


if __name__ == "__main__":
    main()
