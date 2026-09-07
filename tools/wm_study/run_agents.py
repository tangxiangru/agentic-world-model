"""Run the two RPM arms (inference-only vs. +world model) over decision sets.

Each call is a fresh `claude -p` session confined to a workspace; every tool call
is captured and audited for escapes. The agent's decision.json is scored against
the private labels.

Usage:
  python -m tools.wm_study.run_agents --arms base wm --benchmarks gsm8k --limit 2
  python -m tools.wm_study.run_agents --score-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tools.wm_study.decision_sets import materialize_workspace

SYSTEM = """You are an expert ML research allocator deciding which of several proposed
post-training experiments should get the GPU. You reason carefully from evidence:
the candidates' recipes and code, and the outcomes of prior experiments by other
scientists on the same task. Work only inside the current directory, never use the
network, and never try to discover the candidates' true outcomes. Be decisive: you
must end by writing decision.json and replying with that JSON."""

PROMPT = """Read task.md first, then decide. Be efficient: use prior_runs/INDEX.md and the
SUMMARY.md files to find the most comparable prior experiments, open their cards
and code when they matter, and read every candidate's recipe and scripts.{extra}
Write decision.json and reply with the same JSON object as your final message."""

WM_EXTRA = """
Also read wm/README.md and wm/predictions.json; the world model is trained on
exactly the prior runs you can see and is your best summary of that evidence."""

WM2_EXTRA = """
Also read wm/README.md and wm/predictions.json; the world model is trained on
exactly the prior runs you can see and is your best summary of that evidence.
Default to the world model's ranking: its ordering of siblings is right well
above chance even when the predicted gap is small. Override it only with
concrete, specific evidence you can cite (a bug or format mismatch in a
candidate's script, or a prior run with a near-identical recipe whose outcome
contradicts the prediction). Never choose a candidate the model ranks in the
bottom half without such evidence, and say in the rationale whether you
followed or overrode the model and why."""

ESCAPE = re.compile(r"(?:^|[\s'\"=(])(?:/(?!tmp/awm-agent)[A-Za-z]|~|\.\./)|\b(?:curl|wget|pip|ssh|scp|git\s+clone|nc)\b")


def run_one(ws: Path, arm: str, model: str, budget: float, effort: str, timeout: int):
    cmd = [
        "claude", "-p", "--safe-mode", "--strict-mcp-config", "--no-session-persistence",
        "--tools", "Bash,Read,Glob,Grep,Write",
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json", "--verbose",
        "--effort", effort, "--max-budget-usd", str(budget), "--model", model,
        "--system-prompt", SYSTEM,
    ]
    prompt = PROMPT.format(extra={"wm": WM_EXTRA, "wm2": WM2_EXTRA}.get(arm, ""))
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_") and k not in {"CLAUDECODE"}}
    env["HOME"] = os.environ["HOME"]
    t0 = time.time()
    proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, cwd=ws, timeout=timeout, env=env)
    events = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return {"returncode": proc.returncode, "stderr": proc.stderr[-2000:], "elapsed_s": time.time() - t0, "events": events}


def audit(events, ws: Path):
    """Flag tool calls that reference paths outside the workspace or use the network."""
    flags = []
    n_tools = 0
    inside = {str(ws), str(Path(ws).resolve()), "/private" + str(ws) if str(ws).startswith("/tmp") else str(ws)}

    def outside(text):
        for m in re.finditer(r"/(?:Users|home|etc|var|tmp|private)[\w./\-]*", text):
            if not any(m.group().startswith(i) for i in inside):
                return True
        return False

    for ev in events:
        if ev.get("type") != "assistant":
            continue
        for block in (ev.get("message") or {}).get("content") or []:
            if block.get("type") != "tool_use":
                continue
            n_tools += 1
            args = json.dumps(block.get("input"))
            net = re.search(r"\b(?:curl|wget|pip install|ssh|scp|git clone)\b", args)
            if "../" in args or "~/" in args or net or outside(args):
                flags.append({"tool": block.get("name"), "input": args[:300], "kind": "net" if net else ("dotdot" if "../" in args else "path")})
    return {"n_tool_calls": n_tools, "flags": flags}


def parse_decision(events, ws: Path):
    dec = None
    f = ws / "decision.json"
    if f.exists():
        try:
            dec = json.loads(f.read_text())
        except json.JSONDecodeError:
            dec = None
    result = next((e for e in events if e.get("type") == "result"), {})
    if dec is None:
        text = result.get("result") or ""
        m = re.search(r"\{[^{}]*\"choice\"[^{}]*\}", text, re.S)
        if m:
            try:
                dec = json.loads(m.group())
            except json.JSONDecodeError:
                dec = None
    return dec, {"cost_usd": result.get("total_cost_usd"), "num_turns": result.get("num_turns"), "duration_ms": result.get("duration_ms"), "is_error": result.get("is_error"), "subtype": result.get("subtype")}


def score(entry, labels, dec):
    cands = labels[entry["set_id"]]["candidates"]
    ys = {L: c["official_accuracy"] for L, c in cands.items()}
    best = max(ys.values())
    choice = (dec or {}).get("choice")
    if isinstance(choice, str):
        choice = choice.strip().upper()[:1]
    valid = choice in ys
    chosen = ys[choice] if valid else float(np.mean(list(ys.values())))  # invalid -> random expectation
    return {"choice": choice if valid else None, "valid": valid, "chosen_acc": chosen, "oracle_acc": best, "random_acc": float(np.mean(list(ys.values()))), "regret": best - chosen, "top1": float(valid and abs(chosen - best) < 1e-12), "ranking": (dec or {}).get("ranking"), "confidence": (dec or {}).get("confidence")}


def process(entry, arm, args, labels, wm_files):
    out_dir = args.out / "runs" / arm / entry["benchmark"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{entry['set_id']}.json"
    if out_file.exists() and not args.redo:
        prev = json.loads(out_file.read_text())
        if not (args.redo_invalid and (prev["meta"].get("is_error") or not prev["score"]["valid"])):
            return prev
    ws = materialize_workspace(args.sets, entry, arm, args.ws_root, wm_files)
    run = run_one(ws, arm, args.model, args.budget, args.effort, args.timeout)
    dec, meta = parse_decision(run["events"], ws)
    if meta.get("is_error") and any(e.get("type") == "rate_limit_event" and (e.get("rate_limit_info") or {}).get("status") == "rejected" for e in run["events"]):
        raise RuntimeError(f"rate limited on {entry['set_id']} ({arm}); not recording")
    au = audit(run["events"], ws)
    rec = {"set_id": entry["set_id"], "arm": arm, "benchmark": entry["benchmark"], "cell_id": entry["cell_id"], "fold": entry["fold"], "n_candidates": entry["n_candidates"], "decision": dec, "meta": meta, "audit": au, "elapsed_s": run["elapsed_s"], "returncode": run["returncode"], "stderr": run["stderr"], "score": score(entry, labels, dec)}
    (out_dir / f"{entry['set_id']}.events.jsonl").write_text("\n".join(json.dumps(e) for e in run["events"]))
    out_file.write_text(json.dumps(rec, indent=1))
    return rec


def summarize(records):
    by = {}
    for r in records:
        by.setdefault((r["arm"], r["benchmark"]), []).append(r)
    rows = []
    for (arm, bench), rs in sorted(by.items()):
        s = [r["score"] for r in rs]
        rows.append({"arm": arm, "benchmark": bench, "n": len(rs), "valid": sum(x["valid"] for x in s), "chosen_acc": float(np.mean([x["chosen_acc"] for x in s])), "random_acc": float(np.mean([x["random_acc"] for x in s])), "oracle_acc": float(np.mean([x["oracle_acc"] for x in s])), "top1": float(np.mean([x["top1"] for x in s])), "regret": float(np.mean([x["regret"] for x in s])), "cost_usd": float(sum((r["meta"].get("cost_usd") or 0) for r in rs)), "audit_flags": sum(len(r["audit"]["flags"]) for r in rs)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", type=Path, default=Path("data/analysis/wm_study/rpm"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/rpm"))
    ap.add_argument("--ws-root", type=Path, default=Path("/tmp/awm-agent-ws"))
    ap.add_argument("--wm-dir", type=Path, default=Path("data/analysis/wm_study/rpm/wm_tool"))
    ap.add_argument("--arms", nargs="*", default=["base", "wm"])
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sets-filter", nargs="*", default=None)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--redo-invalid", action="store_true", help="re-run only sets whose saved decision is invalid or errored")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    entries = [json.loads(l) for l in (args.sets / "registry.jsonl").read_text().splitlines() if l.strip()]
    entries = [e for e in entries if e["benchmark"] in args.benchmarks]
    if args.sets_filter:
        entries = [e for e in entries if e["set_id"] in args.sets_filter]
    if args.limit:
        entries = entries[: args.limit]
    labels = json.loads((args.sets / "private_labels.json").read_text())
    records = []
    if args.score_only:
        for arm in args.arms:
            for e in entries:
                f = args.out / "runs" / arm / e["benchmark"] / f"{e['set_id']}.json"
                if f.exists():
                    rec = json.loads(f.read_text())
                    ev_file = f.with_suffix(".events.jsonl")
                    if ev_file.exists():
                        events = [json.loads(l) for l in ev_file.read_text().splitlines() if l.strip()]
                        rec["audit"] = audit(events, args.ws_root / arm / e["benchmark"] / e["set_id"])
                    records.append(rec)
    else:
        jobs = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for arm in args.arms:
                for e in entries:
                    wm_files = None
                    if arm.startswith("wm"):
                        d = (args.wm_dir if arm == "wm" else Path(str(args.wm_dir) + "_v2")) / e["benchmark"] / e["set_id"]
                        wm_files = {p.name: p.read_text() for p in d.iterdir()}
                    jobs.append(pool.submit(process, e, arm, args, labels, wm_files))
            for j in as_completed(jobs):
                try:
                    r = j.result()
                except Exception as ex:  # keep going; record failure
                    print("FAILED:", ex, file=sys.stderr)
                    continue
                records.append(r)
                s = r["score"]
                print(f"{r['arm']:4s} {r['benchmark']:8s} {r['set_id']} n={r['n_candidates']} choice={s['choice']} chosen={s['chosen_acc']:.3f} best={s['oracle_acc']:.3f} cost=${(r['meta'].get('cost_usd') or 0):.2f} turns={r['meta'].get('num_turns')} flags={len(r['audit']['flags'])}", flush=True)
    rows = summarize(records)
    for r in rows:
        print(json.dumps(r))
    (args.out / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
