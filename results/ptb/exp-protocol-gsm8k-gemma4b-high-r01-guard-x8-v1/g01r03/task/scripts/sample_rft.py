"""Sample k solutions per training problem from a checkpoint and keep the correct ones.

Problems come from GSM8K *train* and from OpenMathInstruct-2's GSM8K-derived
problems; nothing here touches the GSM8K test split. Prompts are rendered with
the harness's own template so the samples are in the graded distribution.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
GSM8K_TRAIN_GLOB = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"
INT_RE = re.compile(r"^-?\d+$")
LAST_NUM = re.compile(r"-?\d[\d,]*")


def final_answer(text: str) -> str | None:
    """What inspect's match(numeric=True) would read: the last number in the text."""
    m = LAST_NUM.findall(text.replace("$", ""))
    if not m:
        return None
    return m[-1].replace(",", "")


def load_problems(n_gsm: int, n_aug: int, seed: int):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    probs = []
    tr = pq.read_table(sorted(glob.glob(GSM8K_TRAIN_GLOB))[0]).to_pylist()
    for r in tr[:n_gsm]:
        probs.append({"problem": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
                      "source": "gsm8k_train"})
    if n_aug:
        seen = set()
        aug = []
        for f in sorted(glob.glob(OMI2_GLOB)):
            t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pylist()
            for r in t:
                if r["problem_source"] != "augmented_gsm8k":
                    continue
                if r["problem"] in seen or not INT_RE.match(r["expected_answer"].strip()):
                    continue
                seen.add(r["problem"])
                aug.append({"problem": r["problem"], "answer": r["expected_answer"].strip(),
                            "source": "augmented_gsm8k"})
        rng.shuffle(aug)
        probs.extend(aug[:n_aug])
    rng.shuffle(probs)
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=7473)
    ap.add_argument("--n-aug", type=int, default=30000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert render.template_hash() == render.TEMPLATE_SHA256, "gemma3.jinja changed"
    from vllm import LLM, SamplingParams

    probs = load_problems(args.n_gsm, args.n_aug, args.seed)
    print(f"{len(probs)} problems x k={args.k}", flush=True)
    prompts = [render.render_prompt(p["problem"]) for p in probs]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=2048, seed=args.seed, enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept = solved = 0
    with open(args.out, "w") as fh:
        for p, o in zip(probs, outs):
            texts, n_here = [], 0
            got = False
            for c in o.outputs:
                t = c.text.strip()
                if not t or render.ANSWER_MARKER not in t:
                    continue
                if t.count(render.ANSWER_MARKER) != 1:
                    continue
                if final_answer(t.split(render.ANSWER_MARKER)[-1]) != p["answer"]:
                    continue
                got = True
                if t in texts or n_here >= args.max_per_problem:
                    continue
                texts.append(t)
                n_here += 1
                body = t.rsplit(render.ANSWER_MARKER, 1)[0].rstrip()
                fh.write(json.dumps({
                    "prompt": render.render_prompt(p["problem"]),
                    "completion": render.build_completion(body, p["answer"]),
                    "answer": render.format_answer(p["answer"]),
                    "source": "rft:" + p["source"],
                    "nshot": 0,
                }, ensure_ascii=False) + "\n")
                kept += 1
            solved += int(got)
    print(f"kept {kept} samples; {solved}/{len(probs)} problems solved at least once "
          f"({solved/len(probs):.3f})", flush=True)


if __name__ == "__main__":
    main()
