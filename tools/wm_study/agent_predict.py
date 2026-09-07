"""Agent predictor: Claude reads a recipe, the parent's accuracy, and the fold's
training-run bank (recipes with official scores), and predicts the child's official accuracy.

Same folds as the world model; the bank never contains the test session.
Usage: python -m tools.wm_study.agent_predict --benchmarks gsm8k --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tools.wm_study.decision_sets import short_recipe
from tools.wm_study.extract import BENCH_INFO
from tools.wm_study.wm import load_cards, make_folds
from tools.wm_study.wm_delta import parent_acc

BASE = {"gsm8k": 0.045, "aime2025": 0.05}

SYSTEM = """You are an expert in post-training small language models. You will be shown one
planned experiment: the recipe that will be run (every training step from the base model,
with data, method, hyperparameters and launch command), the measured official accuracy of
the checkpoint it starts from, and a bank of earlier experiments by other scientists on the
same task with their recipes and official scores. Predict the official test accuracy the
experiment's checkpoint will get. Reason from the bank and from what the recipe does; think
about training dose, data quality, format, and failure modes seen in the bank. Reply with
exactly one JSON object: {"predicted_accuracy": <number 0-1>, "predicted_change": <number>,
"rationale": "<at most 60 words>"}."""


def compact_setup(setup):
    s = json.loads(json.dumps(setup))
    cmd = s.get("command") or {}
    for k in ("cwd", "env", "log"):
        cmd.pop(k, None)
    for d in s.get("data") or []:
        for k in ("path", "built_by", "build_command"):
            d.pop(k, None)
    s.pop("output_dir", None)
    s.pop("resume_argv", None)
    return s


def build_prompt(card, pacc, bank, benchmark):
    info = BENCH_INFO[benchmark]
    steps = []
    for st in card["recipe"]:
        role = "THIS EXPERIMENT" if st["relation"] == "target" else ("weight ancestor " + st["card_id"] if st["relation"] == "weights" else "data-generation dependency " + st["card_id"])
        steps.append(f"### {role} (family: {st['family']})\n" + json.dumps(compact_setup(st["setup"]), ensure_ascii=False, indent=None))
        if st["relation"] == "target":
            for sc in st.get("scripts") or []:
                steps.append(f"--- script {sc['name']} (first 2500 chars) ---\n{sc['content'][:2500]}")
    bank_lines = ["run | card | parent | parent official acc | official acc | recipe"]
    for b in bank:
        bank_lines.append(f"{b['run']} | {b['card']} | {b['parent']} | {b['pacc']} | {b['acc']:.3f} | {b['recipe']}")
    parent_desc = "the base model" if not card["parents"] else "checkpoint " + ",".join(card["parents"]) + " (a weight ancestor listed below)"
    return f"""# Task
Post-training `{info['base_model']}` for {'GSM8K' if benchmark == 'gsm8k' else 'AIME 2025'}; official test = {info['n_eval']} problems, exact-match accuracy.
The experiment starts from {parent_desc}, whose official accuracy is **{pacc:.3f}**.

# Bank of earlier experiments by other scientists (official scores on the same test set)
{chr(10).join(bank_lines)}

# The planned experiment
{chr(10).join(steps)}

Predict this experiment's official accuracy. Reply with the JSON object only."""


def call(prompt, model, budget, effort, timeout):
    cmd = ["claude", "-p", "--safe-mode", "--strict-mcp-config", "--no-session-persistence", "--tools", "", "--output-format", "json", "--effort", effort, "--max-budget-usd", str(budget), "--model", model, "--system-prompt", SYSTEM]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_") and k != "CLAUDECODE"}
    with tempfile.TemporaryDirectory() as wd:
        proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, cwd=wd, timeout=timeout, env=env)
    out = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {"result": proc.stdout, "is_error": True}
    text = out.get("result") or ""
    m = re.search(r"\{[^{}]*predicted_accuracy[^{}]*\}", text, re.S)
    pred = None
    if m:
        try:
            pred = json.loads(m.group())
        except json.JSONDecodeError:
            pred = None
    return pred, {"cost_usd": out.get("total_cost_usd"), "is_error": out.get("is_error"), "raw": text[:400], "rate_limited": "session limit" in text.lower() or "rate limit" in text.lower()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/agent_predict"))
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k"])
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for bench in args.benchmarks:
        all_rows = load_cards(args.cards, bench, labeled_only=False)
        by_id = {r["example_id"]: r for r in all_rows}
        rows = []
        for r in all_rows:
            if r["official_accuracy"] is None or not r["lineage_complete"]:
                continue
            pa = parent_acc(r, by_id)
            if pa is None:
                if r["parents"]:
                    continue
                pa = BASE[bench]
            r["_pacc"] = pa
            rows.append(r)
        fold_of = make_folds(rows, 8)
        out_dir = args.out / bench
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [r for r in rows if not (out_dir / (r["example_id"].replace("/", "__") + ".json")).exists()]
        if args.limit:
            todo = todo[: args.limit]
        print(f"{bench}: {len(rows)} experiments, {len(todo)} to run", flush=True)

        def bank_for(fold):
            bank = []
            for b in rows:
                if fold_of[b["cell_id"]] == fold:
                    continue
                pb = b["_pacc"]
                bank.append({"run": b["cell_id"], "card": b["card_id"], "parent": ",".join(b["parents"]) or "base", "pacc": f"{pb:.3f}", "acc": b["official_accuracy"], "recipe": short_recipe(b)})
            return sorted(bank, key=lambda x: (x["run"], x["card"]))

        def work(r):
            prompt = build_prompt(r, r["_pacc"], bank_for(fold_of[r["cell_id"]]), bench)
            t0 = time.time()
            pred, meta = call(prompt, args.model, args.budget, args.effort, args.timeout)
            if meta["rate_limited"]:
                raise RuntimeError("rate limited: " + r["example_id"])
            rec = {"example_id": r["example_id"], "fold": fold_of[r["cell_id"]], "parent_acc": r["_pacc"], "true": r["official_accuracy"], "pred": pred, "meta": meta, "elapsed_s": time.time() - t0, "prompt_chars": len(prompt)}
            (out_dir / (r["example_id"].replace("/", "__") + ".json")).write_text(json.dumps(rec, indent=1))
            return rec

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(work, r) for r in todo]
            for f in as_completed(futs):
                try:
                    rec = f.result()
                    p = (rec["pred"] or {}).get("predicted_accuracy")
                    print(f"{rec['example_id']:18s} parent {rec['parent_acc']:.3f} true {rec['true']:.3f} agent {p if p is None else round(float(p), 3)} cost ${rec['meta'].get('cost_usd') or 0:.2f}", flush=True)
                except Exception as e:
                    print("FAILED:", e, flush=True)


if __name__ == "__main__":
    main()
