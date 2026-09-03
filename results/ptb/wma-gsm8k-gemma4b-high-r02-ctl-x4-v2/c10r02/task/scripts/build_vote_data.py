"""Turn k independent samples per question into one training target that makes
the model do self-consistency inside a single graded completion:

    Attempt 1: ... Answer: a1
    Attempt 2: ... Answer: a2
    Attempt 3: ... Answer: a3
    The three attempts give a1, a2, a3. The most frequent answer is G.

    ANSWER: G

Only triples whose modal answer IS the gold answer are kept, and triples that
contain a genuine disagreement are preferred, so the model learns to count
rather than to copy.
"""
import argparse
import collections
import itertools
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/vote_raw.jsonl")
    ap.add_argument("--out", default="data/sft_vote.jsonl")
    ap.add_argument("--rows-per-question", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    stats = collections.Counter()
    n_disagree = 0
    with open(args.out, "w") as fh:
        for line in open(args.raw):
            r = json.loads(line)
            cands, gold = r["cands"], r["answer"]
            stats["questions"] += 1
            if len(cands) < 3:
                stats["too_few_cands"] += 1
                continue
            triples = list(itertools.combinations(range(len(cands)), 3))
            rng.shuffle(triples)
            good = []
            for t in triples:
                preds = [cands[i]["pred"] for i in t]
                mode, cnt = collections.Counter(preds).most_common(1)[0]
                if cnt < 2 or mode != gold:
                    continue
                good.append((t, len(set(preds)) > 1))
            if not good:
                stats["no_gold_majority"] += 1
                continue
            good.sort(key=lambda x: not x[1])  # disagreeing triples first
            for t, dis in good[: args.rows_per_question]:
                order = list(t)
                rng.shuffle(order)
                parts, preds = [], []
                for j, i in enumerate(order, 1):
                    parts.append(f"Attempt {j}:\n{cands[i]['body']}\nAnswer: {cands[i]['pred']}")
                    preds.append(cands[i]["pred"])
                target = ("\n\n".join(parts)
                          + f"\n\nThe three attempts give {preds[0]}, {preds[1]}, {preds[2]}. "
                          + f"The most frequent answer is {gold}.\n\nANSWER: {gold}")
                fh.write(json.dumps({"question": r["question"], "target": target,
                                     "answer": gold, "source": "vote"}) + "\n")
                stats["rows"] += 1
                n_disagree += int(dis)
    stats["rows_with_disagreement"] = n_disagree
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
