#!/usr/bin/env python3
"""exp-06 corpus: round-2 rejection-sampled self-solutions, plus a replay slice.

The replay slice (rows the model has already been trained on once) is there to
stop a short pass over self-data alone from narrowing the model: RFT rows all
come from the 6,973 gsm8k train questions, so on their own they are a much
narrower distribution than anything trained on so far.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmt import ANSWER_MARKER, END_OF_TURN, render_prompt_fast  # noqa: E402
from eval_format import build_system_message, build_user_message  # noqa: E402
from sample_model import grade  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def signature(t: str) -> str:
    return "|".join(re.findall(r"\d+\.?\d*", t))


ANSWER_LINE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)")


def truncate_at_answer(text: str) -> str | None:
    """Cut the sample at the end of its FIRST 'ANSWER: <n>' line.

    vLLM's offline LLM.generate() stops only on the tokenizer eos (<eos>, id 1),
    not on <end_of_turn> (106) which the OpenAI server the grader uses does stop
    on. 25,709 of 27,892 samples therefore ran past a complete answer into junk
    and were scored on whatever number came last. The solution itself is intact
    up to the first ANSWER line, so we keep that and drop the tail.
    """
    m = ANSWER_LINE.search(text)
    if not m:
        return None
    return text[: m.end()].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(DATA, "rft2_samples.jsonl"))
    ap.add_argument("--replay", default=os.path.join(DATA, "sft_v2.jsonl"))
    ap.add_argument("--replay-n", type=int, default=10000)
    ap.add_argument("--max-per-q", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--out", default=os.path.join(DATA, "sft_v4.jsonl"))
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hold = {
        json.loads(l)["question"]
        for l in open(os.path.join(DATA, "dev_gsm8k_trainholdout.jsonl"))
    }

    rft, n_q, n_cov, n_stop, n_tot = [], 0, 0, 0, 0
    for line in open(args.samples):
        r = json.loads(line)
        n_q += 1
        if r["question"] in hold:
            continue
        n_tot += len(r["samples"])
        n_stop += sum(1 for s in r["samples"] if s["finish"] == "stop")
        gold = str(r["gold"]).replace(",", "").strip()
        cands = []
        for s in r["samples"]:
            cut = truncate_at_answer(s["text"])
            if cut is None:
                continue
            _, ok = grade(cut, gold)
            if ok:
                cands.append({"text": cut})
        cands.sort(key=lambda s: len(s["text"]))
        seen, keep = set(), []
        for s in cands:
            t = s["text"].strip()
            if t.count(ANSWER_MARKER) != 1 or len(t) > 2600:
                continue
            sig = signature(t)
            if sig in seen:
                continue
            seen.add(sig)
            keep.append(t)
            if len(keep) >= args.max_per_q:
                break
        if keep:
            n_cov += 1
        for t in keep:
            rft.append({"question": r["question"], "target": t})

    system = build_system_message()
    out = []
    n_few = int(len(rft) * args.fewshot_frac)
    rng.shuffle(rft)
    for i, r in enumerate(rft):
        sysm = system if i < n_few else None
        out.append(
            {
                "prompt": render_prompt_fast(sysm, build_user_message(r["question"])),
                "completion": r["target"].strip() + END_OF_TURN,
                "source": "rft2_self",
                "fewshot": sysm is not None,
            }
        )

    replay = [json.loads(l) for l in open(args.replay)]
    rng.shuffle(replay)
    for e in replay[: args.replay_n]:
        out.append({**e, "source": "replay_sft_v2"})

    rng.shuffle(out)
    with open(args.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")

    print(json.dumps({
        "questions": n_q,
        "samples_total": n_tot,
        "samples_terminated": n_stop,
        "terminated_frac": round(n_stop / max(1, n_tot), 4),
        "questions_covered": n_cov,
        "coverage": round(n_cov / max(1, n_q), 4),
        "rft_rows": len(rft),
        "replay_rows": min(args.replay_n, len(replay)),
        "total": len(out),
        "out": args.out,
    }, indent=1))


if __name__ == "__main__":
    main()
