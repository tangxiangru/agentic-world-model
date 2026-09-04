"""Round-2 supervision: fresh rows for problems the first epoch saw only once.

Epoch 1 deduplicated OpenMathInstruct-2 on the question, so each of its 80,812
GSM8K-source problems contributed exactly ONE of the several distinct solutions
the dataset holds.  This builds the *other* solutions, plus MetaMathQA GSM
rephrasings (fresh question texts, same train provenance) and the Orca-Math rows
epoch 1 did not use.  Same target contract as epoch 1: one 'ANSWER: n' line last,
then <end_of_turn>.
"""
import argparse, hashlib, json, random, re, sys
from datasets import load_dataset

sys.path.insert(0, "work")
from build_sft_data import make_row, clean_solution, last_number, norm_num, PROMPT_TEMPLATE

STOP = "<end_of_turn>"


def key(q):
    return hashlib.md5(q.strip().lower().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v2.jsonl")
    ap.add_argument("--omi-extra", type=int, default=45000)
    ap.add_argument("--metamath", type=int, default=22000)
    ap.add_argument("--orca-extra", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # every (question, target) pair epoch 1 already trained on
    seen_pairs, seen_q = set(), set()
    for l in open("data/sft_v1.jsonl"):
        d = json.loads(l)
        seen_pairs.add(hashlib.md5((d["question"] + "||" + d["target"]).lower().encode()).hexdigest())
        seen_q.add(key(d["question"]))
    dev = {json.loads(l)["question"].strip().lower() for l in
           open("data/dev300_gsm8ktrain.jsonl")}
    print(f"epoch-1 rows: {len(seen_pairs)} unique questions: {len(seen_q)}", flush=True)

    rows = []

    def add(q, sol, ans, src):
        if q.strip().lower() in dev:
            return False
        r = make_row(q, sol, ans)
        if r is None:
            return False
        tgt = r[1] + STOP
        h = hashlib.md5((q.strip() + "||" + tgt).lower().encode()).hexdigest()
        if h in seen_pairs:
            return False
        seen_pairs.add(h)
        rows.append({"prompt": r[0], "target": tgt, "source": src,
                     "question": q.strip(), "fewshot": 0})
        return True

    # --- other OpenMathInstruct-2 solutions to the same GSM8K problems --------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    idx = list(range(len(omi)))
    rng.shuffle(idx)
    n = 0
    for i in idx:
        if n >= a.omi_extra:
            break
        r = omi[i]
        if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
            continue
        if add(r["problem"], r["generated_solution"], r["expected_answer"],
               "omi2_gsm8k_alt"):
            n += 1
    print(f"omi2 alternative solutions: {n}", flush=True)

    # --- MetaMathQA GSM rephrasings ------------------------------------------
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    idx = list(range(len(mm)))
    rng.shuffle(idx)
    m = 0
    for i in idx:
        if m >= a.metamath:
            break
        r = mm[i]
        if not r["type"].startswith("GSM"):
            continue
        resp = r["response"].strip()
        mt = re.search(r"The answer is:?\s*(.+)$", resp)
        if not mt:
            continue
        ans = norm_num(mt.group(1))
        if ans is None:
            continue
        if add(r["query"], resp, ans, "metamath_gsm"):
            m += 1
    print(f"metamath GSM: {m}", flush=True)

    # --- Orca-Math rows epoch 1 did not use ----------------------------------
    orca = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    idx = list(range(len(orca)))
    rng.shuffle(idx)
    o = 0
    for i in idx:
        if o >= a.orca_extra:
            break
        r = orca[i]
        if key(r["question"]) in seen_q:
            continue
        ans = last_number(r["answer"])
        if ans is None:
            continue
        if add(r["question"], r["answer"], ans, "orca_extra"):
            o += 1
    print(f"orca fresh: {o}", flush=True)

    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
