"""Build the SFT corpus from nvidia/OpenMathInstruct-2 (+ gsm8k train few-shot
contexts), formatted for the grader: gemma3 chat turns, one 'ANSWER: N' line at
the end, terminated by <end_of_turn>.

No GSM8K *test* item is read anywhere in this file.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OMI2 = sorted(
    glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
    )
)
GSM_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
)[0]

GSM_SOURCES = {"gsm8k", "augmented_gsm8k"}
MATH_SOURCES = {"math", "augmented_math"}


HEAD = 'is the answer to the problem.\n\n'
TAIL = '\n\nRemember to put your answer'


def extract_question(rec: dict) -> str:
    """Recover the raw problem text from a built row's rendered prompt."""
    p = rec.get("prompt", "")
    i = p.rfind(HEAD)
    j = p.find(TAIL, i)
    return p[i + len(HEAD) : j].strip() if i >= 0 and j > i else ""


def norm_key(s: str) -> str:
    return hashlib.md5(re.sub(r"\W+", "", s.lower()).encode()).hexdigest()


def load_gsm_train() -> list[dict]:
    t = pq.read_table(GSM_TRAIN).to_pylist()
    out = []
    for r in t:
        q = r["question"].strip()
        a = r["answer"]
        reasoning, _, target = a.partition("####")
        out.append(
            {
                "question": q,
                "reasoning": common.strip_calc(reasoning).strip(),
                "answer": target.strip().replace(",", ""),
            }
        )
    return out


def make_fewshot_block(pool: list[dict], k: int, rng: random.Random) -> str:
    picks = rng.sample(pool, k)
    return "\n\n".join(
        common.fewshot_example(p["question"], p["reasoning"], p["answer"]) for p in picks
    )


JUNK = (
    "not enough information",
    "cannot be determined",
    "can't provide",
    "cannot provide",
    "doesn't allow",
    "does not allow",
    "i will provide",
    "the problem as stated",
    "please note that",
    "i'll provide",
    "seems to be a mistake",
    "there is a mistake",
)

# trailing "The final answer is: 42" / "The answer is 42." style sign-offs
TAIL_ANS = re.compile(
    r"(?:\n|^)\s*(?:so,?\s+|thus,?\s+|therefore,?\s+)?the\s+(?:final\s+)?answer\s+is:?\s*\$?[^\n]{0,40}$",
    re.I,
)


def build_target(solution: str, answer: str) -> str | None:
    if answer not in solution:
        return None
    low = solution.lower()
    if any(j in low for j in JUNK):
        return None
    sol = common.strip_boxed(solution).strip()
    if not sol:
        return None
    # drop any stray gsm8k-style '#### N' marker so only one answer marker remains
    sol = re.sub(r"\n?####\s*[-\d,\.]+\s*$", "", sol).strip()
    # drop a trailing "The final answer is ..." sign-off: the ANSWER: line replaces it
    sol = TAIL_ANS.sub("", sol).strip()
    if len(sol) < 40 or "ANSWER:" in sol:
        return None
    return f"{sol}\n\nANSWER: {answer}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=90_000)
    ap.add_argument("--n-math", type=int, default=30_000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--max-prob-chars", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--exclude", default=None, help="comma-separated jsonl files whose (question,solution) pairs must not reappear")
    ap.add_argument("--exclude-problems", type=int, default=0, help="1 = also exclude the problems entirely")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm_pool = load_gsm_train()
    print(f"gsm8k train pool: {len(gsm_pool)}", flush=True)

    seen: dict[str, int] = {}
    used_pairs: set[str] = set()
    for path_ in (args.exclude or "").split(",") if args.exclude else []:
        n0 = 0
        for line in open(path_):
            r = json.loads(line)
            q = r.get("question") or extract_question(r)
            if q:
                if args.exclude_problems:
                    seen[norm_key(q)] = 10**6
                used_pairs.add(norm_key(q + "|" + r["completion"]))
            n0 += 1
        print(f"excluded {n0} rows of {path_} (problem-level={bool(args.exclude_problems)})", flush=True)
    buckets: dict[str, list[dict]] = {"gsm": [], "math": []}
    need = {"gsm": args.n_gsm, "math": args.n_math}
    stats = {"rows": 0, "bad_answer": 0, "too_long": 0, "dup": 0, "bad_target": 0}

    for f in OMI2[args.shard_start:]:
        if all(len(buckets[b]) >= need[b] for b in buckets):
            break
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20_000):
            rows = batch.to_pylist()
            rng.shuffle(rows)
            for r in rows:
                stats["rows"] += 1
                src = r["problem_source"]
                bucket = "gsm" if src in GSM_SOURCES else ("math" if src in MATH_SOURCES else None)
                if bucket is None or len(buckets[bucket]) >= need[bucket]:
                    continue
                ans = common.clean_number(r["expected_answer"] or "")
                if ans is None:
                    stats["bad_answer"] += 1
                    continue
                prob = (r["problem"] or "").strip()
                sol = (r["generated_solution"] or "").strip()
                if len(prob) > args.max_prob_chars or len(sol) > args.max_sol_chars or len(sol) < 40:
                    stats["too_long"] += 1
                    continue
                k = norm_key(prob)
                if seen.get(k, 0) >= args.max_per_problem:
                    stats["dup"] += 1
                    continue
                tgt = build_target(sol, ans)
                if tgt is None:
                    stats["bad_target"] += 1
                    continue
                if used_pairs and norm_key(prob + "|" + common.render_completion(tgt)) in used_pairs:
                    stats["dup"] += 1
                    continue
                seen[k] = seen.get(k, 0) + 1
                buckets[bucket].append(
                    {"question": prob, "target": tgt, "answer": ans, "source": src}
                )
            if all(len(buckets[b]) >= need[b] for b in buckets):
                break
        print(
            f"  {Path(f).name}: gsm={len(buckets['gsm'])} math={len(buckets['math'])}",
            flush=True,
        )

    data = buckets["gsm"] + buckets["math"]
    rng.shuffle(data)
    print("kept:", {k: len(v) for k, v in buckets.items()}, "stats:", stats, flush=True)

    n_fs = int(len(data) * args.fewshot_frac)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for i, d in enumerate(data):
            if i < n_fs:
                k = rng.choice([2, 2, 4, 4, 8, 10])
                fs = make_fewshot_block(gsm_pool, k, rng)
            else:
                fs = None
            rec = {
                "prompt": common.render_prompt(common.user_prompt(d["question"], fs)),
                "completion": common.render_completion(d["target"]),
                "answer": d["answer"],
                "source": d["source"],
                "nshot": 0 if fs is None else k,
                # plain text of the trainable content, for the contamination checker
                "question": d["question"],
                "text": d["question"] + "\n\n" + d["target"],
            }
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(data)} rows -> {out} ({n_fs} with a few-shot context)")


if __name__ == "__main__":
    main()
