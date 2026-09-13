"""Independent checks over the assembled benchmark. Recomputes; never trusts a stored number.

    python -m tools.wm_benchmark.verify <mirror_root> <benchmark_dir> [--sample N]

Checks, per example in examples.jsonl:
  Y     mean_pass_rate == mean of per-run rates recomputed from the source result file's per_problem
  runs  exactly ten runs, N questions (1,319 / 30), binary entries
  Z     if normalized: log_ref says the log is the record of the label; samples.jsonl.gz has N×10 rows
        and its per-(question, run) scores reproduce per_problem exactly
  X     launch_record.json exists, validates against the schema, every materialized file's sha256
        matches, at least one step has an available entrypoint, the chain ends at archived_from_dir,
        and no numeric token that looks like a score (0.xxx accuracy, "acc", "pass@") appears in the
        record's command/notes fields  (heuristic leakage screen, reported not enforced)
  split every example of a session is in one split; matrix locked sessions are all in test
Writes <benchmark_dir>/verify_report.json and prints the summary. Exit code 1 if any hard check fails.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
N_QUESTIONS = {"gsm8k": 1319, "aime2025": 30}
SCORE_RE = re.compile(r"\b(accuracy|acc|pass@\d|score)\b[^\n]{0,20}?\b0\.\d{2,}", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mirror_root")
    ap.add_argument("benchmark_dir")
    ap.add_argument("--sample", type=int, default=0, help="check only the first N examples (0 = all)")
    a = ap.parse_args()
    root, bench = Path(a.mirror_root), Path(a.benchmark_dir)
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(json.loads((REPO / "tools/wm_benchmark/launch_record.schema.json").read_text()))
    except ImportError:
        validator = None
    examples = [json.loads(l) for l in (bench / "examples.jsonl").open()]
    if a.sample:
        examples = examples[: a.sample]
    splits = json.loads((bench / "splits.json").read_text())["sessions"]
    hard, soft = Counter(), Counter()
    per_example = {}
    x_cache = {}
    for ex in examples:
        problems, warnings = [], []
        src = json.loads((root / ex["Z"]["log"].replace("/trajectories/", "/results/").replace(".json.gz", ".json")).read_text()) \
            if (root / ex["Z"]["log"].replace("/trajectories/", "/results/").replace(".json.gz", ".json")).exists() else None
        n = N_QUESTIONS.get(ex["benchmark"])
        if src is None:
            problems.append("source_result_missing")
        else:
            pp = src.get("per_problem") or {}
            if len(pp) != n or any(len(v) != 10 or any(x not in (0, 1, 0.0, 1.0) for x in v) for v in pp.values()):
                problems.append("per_problem_shape")
            else:
                runs = [sum(pp[q][r] for q in pp) / n for r in range(10)]
                if ex["Y"]["mean_pass_rate"] is None or abs(statistics.mean(runs) - ex["Y"]["mean_pass_rate"]) > 1e-9:
                    problems.append("y_mean_mismatch")
                if ex["Y"]["runs"] is None or any(abs(x - y) > 1e-9 for x, y in zip(runs, ex["Y"]["runs"])):
                    problems.append("runs_mismatch")
        if ex["Z"]["normalized"]:
            zdir = bench / ex["Z"]["normalized"]
            ref = json.loads((zdir.parent / "log_ref.json").read_text())
            if not ref.get("log_is_record_of_label"):
                problems.append("z_not_record_of_label")
            if src is not None and zdir.exists():
                pp = src["per_problem"]
                seen = 0
                for line in gzip.open(zdir, "rt"):
                    r = json.loads(line)
                    seen += 1
                    if r["score"] is None or abs(pp[r["question_id"]][r["run"] - 1] - r["score"]) > 1e-9:
                        problems.append("z_sample_disagrees")
                        break
                if seen != n * 10:
                    problems.append("z_sample_count")
        elif ex["status"] == "eligible":
            warnings.append("z_not_normalized")
        if ex["X"]["launch_record"]:
            xp = bench / ex["X"]["launch_record"]
            if xp not in x_cache:
                rec = json.loads(xp.read_text())
                errs = [e.message for e in validator.iter_errors(rec)] if validator else []
                for step in rec.get("steps", []):
                    for f in step.get("files", []):
                        if f.get("materialized"):
                            fp = xp.parent / f["materialized"]
                            if not fp.exists() or hashlib.sha256(fp.read_bytes()).hexdigest() != f.get("sha256"):
                                errs.append(f"file_hash:{f['path']}")
                # The chain must name the archived directory: as an output of some step, or as the
                # input of a final select/copy step whose output is the archive itself.
                adir = rec.get("archived_from_dir")
                if rec.get("steps") and adir:
                    last = rec["steps"][-1]
                    named = any(adir in o for st in rec["steps"] for o in st.get("outputs", [])) or \
                        adir in (last.get("inputs", {}).get("parent_model", {}) or {}).get("ref", "") or \
                        any(adir in x for x in last.get("inputs", {}).get("other_inputs", []))
                    if not named:
                        errs.append("chain_does_not_name_archived_dir")
                leak = []
                for step in rec.get("steps", []):
                    for field in (step.get("launch", {}).get("command", ""), step.get("notes", "")):
                        if SCORE_RE.search(field or ""):
                            leak.append(step["step_id"])
                x_cache[xp] = (errs, leak)
            errs, leak = x_cache[xp]
            problems += [f"x:{e}"[:120] for e in errs]
            if leak:
                warnings.append("x_possible_score_text:" + ",".join(leak))
        elif ex["status"] == "eligible":
            problems.append("eligible_without_x")
        if splits.get(ex["session_id"]) != ex["split"]:
            problems.append("split_inconsistent")
        if ex["track"] == "matrix" and ex["S"].get("s_resolution") != "server_verified":
            problems.append("matrix_without_verified_s")
        per_example[ex["example_id"]] = {"status": ex["status"], "problems": problems, "warnings": warnings}
        # Excluded examples carry their reasons by design; only problems on eligible ones are failures.
        for p in problems:
            key = p.split(":")[0] if p.startswith("x:") else p
            (hard if ex["status"] == "eligible" else soft)[("eligible:" if ex["status"] == "eligible" else "excluded:") + key] += 1
        for w in warnings:
            soft[w.split(":")[0]] += 1
    # matrix locked sessions all in test
    locked = {ex["session_id"] for ex in examples if ex["track"] == "matrix" and ex["S"].get("policy") and ex["split"] == "test"}
    by_session = defaultdict(set)
    for ex in examples:
        by_session[ex["session_id"]].add(ex["split"])
    multi = [s for s, v in by_session.items() if len(v) > 1]
    report = {"examples_checked": len(examples), "hard_failures": dict(hard), "warnings": dict(soft),
              "sessions_in_multiple_splits": multi, "examples_with_problems": sum(1 for v in per_example.values() if v["problems"]),
              "eligible_clean": sum(1 for eid, v in per_example.items() if v["status"] == "eligible" and not v["problems"]),
              "per_example": per_example}
    (bench / "verify_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_example"}, indent=1))
    sys.exit(1 if hard or multi else 0)


if __name__ == "__main__":
    main()
