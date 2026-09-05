#!/usr/bin/env python3
"""Mix the exp-04 corpus: on-policy where the model can already solve, teacher
where it cannot.

For each of the 30,000 gsm8k-derived problems exp-02 has never trained on:
  * if rejection sampling at T=0.8 produced at least one verified-correct
    solution, use up to 2 of the model's own solutions (on-policy);
  * if all 4 samples were wrong, the problem is a measured weak spot - use up
    to 2 OpenMathInstruct-2 teacher solutions for it instead.

Both kinds of row are already in the {prompt, completion, fewshot} schema
train_sft.py reads, with completions terminated by <end_of_turn>.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

SPLIT = (
    'Solve the following math problem step by step. The last line of your response '
    'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    "answer to the problem.\n\n"
)
N_SAMPLED = 30000
OUT = "data/sft_exp04.jsonl"


def question_of(prompt: str) -> str | None:
    if SPLIT not in prompt:
        return None
    return prompt.split(SPLIT, 1)[1].split("\n\nRemember to put your answer")[0]


def main() -> None:
    rng = random.Random(0)

    sampled = [json.loads(l) for l in open("data/rft_problems_unseen.jsonl")][:N_SAMPLED]
    sampled_q = [r["question"] for r in sampled]

    rft = [json.loads(l) for l in open("data/rft_raw.jsonl")]
    solved = set()
    for r in rft:
        q = question_of(r["prompt"])
        if q is not None:
            solved.add(q)
    unsolved = [q for q in sampled_q if q not in solved]
    print(f"sampled problems {len(sampled_q)}, solved {len(solved)}, unsolved {len(unsolved)}")

    unsolved_set = set(unsolved)
    teacher: dict[str, list[dict]] = defaultdict(list)
    lines = open("data/sft_gsm_clean.jsonl").readlines()
    for line in lines[110000:]:
        r = json.loads(line)
        q = question_of(r["prompt"])
        if q in unsolved_set and len(teacher[q]) < 2:
            teacher[q].append(r)
    teacher_rows = [r for rows in teacher.values() for r in rows]
    print(f"teacher rows for weak spots: {len(teacher_rows)} over {len(teacher)} problems")

    rows = rft + teacher_rows
    rng.shuffle(rows)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = {
        "rft_rows": len(rft),
        "teacher_rows": len(teacher_rows),
        "total": len(rows),
        "problems_sampled": len(sampled_q),
        "problems_solved": len(solved),
        "problems_unsolved": len(unsolved),
        "fewshot_rows": sum(r["fewshot"] for r in rows),
    }
    print(json.dumps(stats, indent=2))
    json.dump(stats, open("data/sft_exp04.stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
