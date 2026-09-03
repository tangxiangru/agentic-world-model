"""Turn sampled-and-verified solutions into an RFT training file.

Keeps at most --max-per-problem solutions per question, deduplicated by the
multiset of arithmetic expressions they contain (reasoning-path diversity, as
in rejection-sampling fine-tuning) rather than by raw string equality.
Optionally mixes in a slice of the stage-1 SFT file to limit drift.
"""
import argparse
import json
import random
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
import render  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def norm(a):
    a = str(a).strip().replace(",", "").replace("$", "")
    try:
        f = float(a)
        if f != f or f in (float("inf"), float("-inf")):
            return a
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return a


def eq_signature(text):
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    ops = re.findall(r"[-+*/]", text)
    return (tuple(nums), tuple(ops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-per-easy", type=int, default=1, help="cap for problems solved in every sample; they carry little signal")
    ap.add_argument("--max-chars", type=int, default=2600)
    ap.add_argument("--mix-sft", default=None)
    ap.add_argument("--mix-n", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = render.get_tokenizer()

    import glob

    import pyarrow.parquet as pq

    pool = []
    for r in pq.read_table(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
        )[0]
    ).to_pylist():
        body, _, ans = r["answer"].rpartition("####")
        pool.append((r["question"].strip(), body.strip(), ans.strip().replace(",", "")))

    def fewshot_prefix():
        k = rng.choices([1, 2, 3, 5, 10], weights=[30, 30, 20, 12, 8])[0]
        return "\n\n".join(
            f"{q}\n\nReasoning:\n{b}\n\nANSWER: {a}" for q, b, a in rng.sample(pool, k)
        )

    rows = []
    n_prob = n_sol = 0
    for line in open(args.samples):
        r = json.loads(line)
        if not r["solutions"]:
            continue
        n_prob += 1
        seen = set()
        cap = args.max_per_easy if r["n_correct"] >= r["k"] else args.max_per_problem
        cands = sorted(r["solutions"], key=len)
        picked = []
        for s in cands:
            body = s.strip()
            if len(body) > args.max_chars or len(body) < 20:
                continue
            # cut at the FIRST answer line: a sampled solution sometimes repeats
            # 'ANSWER: N' (twice, occasionally 60+ times in a degenerate loop),
            # and keeping them would teach the model that second marker.
            i = body.find("ANSWER:")
            if i == -1:
                continue
            head = body[: i + len("ANSWER:")]
            first = body[i + len("ANSWER:") :].split("\n")[0]
            body = head + first
            got = NUM.findall(first)
            if not got or norm(got[-1]) != norm(r["gold"]):
                continue
            if body.count("ANSWER:") != 1:
                continue
            sig = eq_signature(body)
            if sig in seen:
                continue
            seen.add(sig)
            picked.append(body)
            if len(picked) >= cap:
                break
        for body in picked:
            n_sol += 1
            rows.append({"id": f"rft-{n_sol}", "source": "synthetic:self", "question": r["question"], "text": body, "answer": r["gold"]})

    print(f"problems with >=1 correct: {n_prob}, kept solutions: {n_sol}", flush=True)

    if args.mix_sft and args.mix_n:
        mix = [json.loads(l) for l in open(args.mix_sft)]
        rng.shuffle(mix)
        for r in mix[: args.mix_n]:
            rows.append(
                {
                    "id": "mix-" + r["id"],
                    "source": "sft_v2",
                    "question": r["question"],
                    "prompt": r["prompt"],
                    "completion": r["completion"],
                    "answer": r["answer"],
                }
            )
        print("mixed in", min(args.mix_n, len(mix)), "stage-1 rows", flush=True)

    rng.shuffle(rows)
    with open(args.out, "w") as fh:
        for r in rows:
            if "prompt" not in r:
                sysmsg = fewshot_prefix() if rng.random() < args.fewshot_frac else None
                r["prompt"] = render.render_prompt(tok, r["question"], system=sysmsg)
                # the sampled text already ends with its ANSWER line; add the stop token
                r["completion"] = r.pop("text").strip() + render.STOP_TOKEN
            fh.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "->", args.out)


if __name__ == "__main__":
    main()
