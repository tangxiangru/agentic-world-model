"""Step 4 of the construction log: join launch records (X) to labels (Y, S) into examples.

    python -m tools.wm_benchmark.assemble <benchmark_dir> <timeline_dir>

Reads  <benchmark_dir>/labels.jsonl                     (step 2)
       <benchmark_dir>/x_raw/<cell>/<checkpoint>.json    (step 3, extractor output)
       <benchmark_dir>/x_verify/<cell>/<checkpoint>.json (step 3, verifier output; optional)
       <timeline_dir>/_files/<sha256>                    (file contents cited by the records)
Writes <benchmark_dir>/x/<checkpoint>/launch_record.json   the record, with every cited file
                                                            copied under files/<step_id>/<path>
       <benchmark_dir>/examples.jsonl   one row per (checkpoint, serving) with X pointer, S, Y, Z pointer,
                                        status and every reason an example is not release-eligible
       <benchmark_dir>/splits.json      session-level split: matrix locked sessions → test,
                                        the rest → train/validation by a salted hash of the session
       <benchmark_dir>/assemble_summary.json

An example is `eligible` only when its label is valid, its launch record exists, validates against
the schema, has no `unavailable` entrypoint, the verifier (if run) did not return `needs_fix`, and
its Z log (if normalized) is the record of the label. Everything else stays in the table with its
reasons, so a reader can see what was excluded and why.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import jsonschema
except ImportError:  # validated in a venv with jsonschema; without it we only check required keys
    jsonschema = None

SALT = "awm-wm-benchmark-v2-2026-09-12"
REPO = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def validate_record(record: dict, schema: dict) -> list[str]:
    if jsonschema is not None:
        v = jsonschema.Draft202012Validator(schema)
        return [f"{'/'.join(str(p) for p in e.absolute_path)}: {e.message}"[:200] for e in v.iter_errors(record)]
    return [f"missing:{k}" for k in schema["required"] if k not in record]


def materialize(record: dict, cell: str, x_dir: Path, files_dir: Path, raw_dir: Path) -> list[str]:
    """Copy every cited file's launch-time content next to the record; return problems."""
    problems = []
    for step in record.get("steps", []):
        for f in step.get("files", []):
            src = f.get("source", "")
            dest = x_dir / "files" / step["step_id"].replace("/", "_") / Path(f["path"]).name
            content_file = f.get("content_file")
            candidates = []
            if f.get("sha256"):
                candidates.append(files_dir / f["sha256"])
            if content_file:
                cf = Path(content_file)
                candidates.append(cf if cf.is_absolute() else REPO / cf)
                candidates.append(raw_dir / "files" / cf.name)
            found = next((c for c in candidates if c.exists()), None)
            if src.startswith("unavailable"):
                continue
            if found is None:
                problems.append(f"{step['step_id']}:{f['path']}: content not found ({src})")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, dest)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            if f.get("sha256") and digest != f["sha256"]:
                problems.append(f"{step['step_id']}:{f['path']}: sha mismatch")
            f["materialized"] = str(dest.relative_to(x_dir))
            f["sha256"] = f.get("sha256") or digest
    return problems


def main(argv):
    bench, tl = Path(argv[1]), Path(argv[2])
    schema = json.loads((REPO / "tools/wm_benchmark/launch_record.schema.json").read_text())
    labels = load_jsonl(bench / "labels.jsonl")
    by_ckpt = defaultdict(list)
    for row in labels:
        by_ckpt[row["checkpoint_id"]].append(row)
    records, problems = {}, {}
    for path in sorted((bench / "x_raw").glob("*/*.json")):
        cell = path.parent.name
        rec = json.loads(path.read_text())
        cid = rec.get("checkpoint_id", path.stem)
        errs = validate_record(rec, schema)
        x_dir = bench / "x" / cid
        x_dir.mkdir(parents=True, exist_ok=True)
        errs += materialize(rec, cell, x_dir, tl / "_files", path.parent)
        entry_missing = [s["step_id"] for s in rec.get("steps", [])
                         if not any(f.get("role") == "entrypoint" and not f.get("source", "").startswith("unavailable")
                                    for f in s.get("files", []))]
        if entry_missing:
            errs.append("steps_without_available_entrypoint:" + ",".join(entry_missing))
        ver = bench / "x_verify" / cell / f"{cid}.json"
        verdict = json.loads(ver.read_text()).get("verdict") if ver.exists() else None
        rec["_verification"] = {"verdict": verdict, "path": str(ver.relative_to(bench)) if ver.exists() else None}
        (x_dir / "launch_record.json").write_text(json.dumps(rec, indent=1))
        records[cid], problems[cid] = rec, errs
    # splits: session level
    sessions = {}
    for row in labels:
        s = row["session_id"]
        if row.get("matrix_split") == "locked_session_test":
            sessions[s] = "test"
        else:
            sessions.setdefault(s, None)
    for s, v in sessions.items():
        if v is None:
            bucket = int(hashlib.sha256((SALT + s).encode()).hexdigest()[:8], 16) % 5
            sessions[s] = "validation" if bucket == 0 else "train"
    examples, status = [], Counter()
    for row in labels:
        cid = row["checkpoint_id"]
        rec = records.get(cid)
        reasons = []
        if row["status"] != "valid":
            reasons += [f"label:{e}" for e in row["errors"] if not e.startswith("sample_errors_unknown")]
        if rec is None:
            reasons.append("x:no_launch_record")
        else:
            reasons += [f"x:{e}" for e in problems[cid]]
            if rec["_verification"]["verdict"] == "needs_fix":
                reasons.append("x:verifier_needs_fix")
            if rec.get("confidence") == "low":
                reasons.append("x:low_confidence")
        zref = bench / "z" / row["example_id"] / "log_ref.json"
        z = json.loads(zref.read_text()) if zref.exists() else None
        if z is not None and not z.get("log_is_record_of_label"):
            reasons.append("z:log_not_record_of_label")
        ex = {
            "example_id": row["example_id"], "checkpoint_id": cid, "serving_id": row["serving_id"],
            "session_id": row["session_id"], "split": sessions[row["session_id"]],
            "benchmark": row["benchmark"], "base_model": row["base_model"], "track": row["track"],
            "status": "eligible" if not reasons else "excluded", "reasons": reasons,
            "X": {"launch_record": f"x/{cid}/launch_record.json" if rec else None,
                  "chain_steps": len(rec["steps"]) if rec else None,
                  "confidence": rec.get("confidence") if rec else None,
                  "verification": rec["_verification"]["verdict"] if rec else None},
            "S": row["S"], "Y": {"mean_pass_rate": row["y_mean"], "runs": row["runs"], "run_sd": row["run_sd"]},
            "Z": {"log": row["z_log"]["path"], "normalized": f"z/{row['example_id']}/samples.jsonl.gz" if z else None,
                  "log_sha256": z.get("log_sha256") if z else None},
            "weights": row["weights"], "aggregates": row["aggregates"],
        }
        examples.append(ex)
        status[(row["track"], ex["status"])] += 1
    with (bench / "examples.jsonl").open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, sort_keys=True) + "\n")
    (bench / "splits.json").write_text(json.dumps({"salt": SALT, "rule": "matrix locked_session_test sessions -> test; others sha256(salt+session)%5==0 -> validation else train",
                                                   "sessions": sessions}, indent=1))
    reason_counts = Counter(r.split(":")[0] + ":" + r.split(":")[1] if ":" in r else r for ex in examples for r in ex["reasons"])
    summary = {"examples": len(examples), "by_track_status": {f"{k[0]}/{k[1]}": v for k, v in sorted(status.items())},
               "checkpoints_with_record": len(records), "records_with_problems": sum(1 for v in problems.values() if v),
               "reasons": dict(reason_counts.most_common()),
               "eligible_by_split": dict(Counter(ex["split"] for ex in examples if ex["status"] == "eligible")),
               "eligible_by_benchmark": dict(Counter(ex["benchmark"] for ex in examples if ex["status"] == "eligible"))}
    (bench / "assemble_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv)
