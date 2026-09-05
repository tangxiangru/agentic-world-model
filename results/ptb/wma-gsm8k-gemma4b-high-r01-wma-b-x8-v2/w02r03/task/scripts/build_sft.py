"""Build the SFT jsonl from OpenMathInstruct-2 (gsm8k-sourced) + GSM8K train.

Output rows: {"id", "source", "question", "prompt", "completion", "answer"}
- prompt     : exactly what the grader will send (templates/gemma3.jinja)
- completion : reasoning body, one "ANSWER: N" line, then <end_of_turn>

Nothing here touches the GSM8K *test* split. 500 GSM8K train problems are
held out as a local dev set and excluded from training.
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
import render  # noqa: E402

RAW = sorted(glob.glob("/home/ben/task/data/raw/omi2_*.jsonl"))
GSM_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
)[0]

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def clean_body(sol: str) -> str | None:
    s = sol.strip()
    # unwrap \boxed{...} -> ...  (one answer marker only; pitfall double_answer_format)
    prev = None
    while prev != s:
        prev = s
        s = BOXED.sub(r"\1", s)
    if "\\boxed" in s or "boxed{" in s:
        return None
    if "ANSWER:" in s.upper():
        return None
    if "####" in s:
        return None
    return s.strip()


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if not NUMLIKE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    if "." in a:
        a = a.rstrip("0").rstrip(".")
    return a or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-omi2", type=int, default=140000)
    ap.add_argument("--gsm-train-repeat", type=int, default=2)
    ap.add_argument("--max-body-chars", type=int, default=2600)
    ap.add_argument("--out", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--dev-out", default="/home/ben/task/data/devtrain500.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = render.get_tokenizer()

    # ---- GSM8K train, native style (matches the 10-shot examples the grader uses)
    import pyarrow.parquet as pq

    gsm = pq.read_table(GSM_TRAIN).to_pylist()
    rng.shuffle(gsm)
    dev = gsm[:500]
    gsm_tr = gsm[500:]
    dev_qs = {r["question"].strip() for r in dev}
    with open(args.dev_out, "w") as fh:
        for i, r in enumerate(dev):
            ans = r["answer"].split("####")[-1].strip().replace(",", "")
            fh.write(
                json.dumps({"id": f"devtrain-{i}", "question": r["question"], "gold": ans})
                + "\n"
            )
    print(f"held out {len(dev)} gsm8k-train problems -> {args.dev_out}", flush=True)

    rows = []
    for rep in range(args.gsm_train_repeat):
        for i, r in enumerate(gsm_tr):
            body, _, ans = r["answer"].rpartition("####")
            ans = norm_answer(ans)
            body = body.strip()
            if ans is None or not body:
                continue
            rows.append(
                {
                    "id": f"gsm8ktrain-{i}-r{rep}",
                    "source": "gsm8k_train",
                    "question": r["question"].strip(),
                    "body": body,
                    "answer": ans,
                }
            )
    print("gsm8k train rows:", len(rows), flush=True)

    # ---- OpenMathInstruct-2, gsm8k-sourced
    per_problem: dict[str, int] = {}
    omi = []
    files = RAW[:]
    rng.shuffle(files)
    for path in files:
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                q = r["problem"].strip()
                if q in dev_qs:
                    continue
                if per_problem.get(q, 0) >= args.max_per_problem:
                    continue
                ans = norm_answer(r["expected_answer"])
                if ans is None:
                    continue
                body = clean_body(r["generated_solution"])
                if not body or len(body) > args.max_body_chars or len(body) < 20:
                    continue
                per_problem[q] = per_problem.get(q, 0) + 1
                omi.append(
                    {
                        "id": f"omi2-{len(omi)}",
                        "source": r["problem_source"],
                        "question": q,
                        "body": body,
                        "answer": ans,
                    }
                )
        if len(omi) >= args.n_omi2 * 1.05:
            break
    rng.shuffle(omi)
    omi = omi[: args.n_omi2]
    print("omi2 rows:", len(omi), "unique problems:", len(per_problem), flush=True)

    rows.extend(omi)
    rng.shuffle(rows)

    n = 0
    with open(args.out, "w") as fh:
        for r in rows:
            prompt = render.render_prompt(tok, r["question"])
            completion = render.render_target(r["body"], r["answer"])
            fh.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "source": r["source"],
                        "question": r["question"],
                        "prompt": prompt,
                        "completion": completion,
                        "answer": r["answer"],
                    }
                )
                + "\n"
            )
            n += 1
    print("wrote", n, "->", args.out)


if __name__ == "__main__":
    main()
