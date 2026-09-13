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
import re
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


PTB_TEMPLATES = REPO / "third_party/PostTrainBench/src/eval/templates"


def _ptb_template(path: str):
    """The scientists' `templates/<name>.jinja` are PostTrainBench's own files, unchanged: every
    copy recovered from a trace hashes to the submodule's file (modulo a trailing newline lost
    in `cat` output). Resolve them from the submodule when the record cites one."""
    if "templates/" in path and path.endswith(".jinja"):
        cand = PTB_TEMPLATES / Path(path).name
        return cand if cand.exists() else None
    return None


def materialize(record: dict, cell: str, x_dir: Path, files_dir: Path, raw_dir: Path) -> list[str]:
    """Copy every cited file's launch-time content next to the record; return problems."""
    problems = []
    for step in record.get("steps", []):
        launch_seq = (step.get("launch") or {}).get("seq")
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
            # An inline script (`python -c`, `python - <<EOF`, a shell one-liner) that is the
            # launch command itself has its content in launch.command; no separate copy is needed.
            m = re.match(r"(inline|heredoc)@seq=(\d+)$", src)
            if found is None and m and launch_seq is not None and int(m.group(2)) == launch_seq \
                    and f.get("role") == "inline_script" and (step.get("launch") or {}).get("command"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest = dest.with_name(dest.name + ".launch_command.txt")
                dest.write_text(step["launch"]["command"])
                f["materialized"] = str(dest.relative_to(x_dir))
                f["materialized_from"] = "launch.command"
                continue
            ptb = _ptb_template(f["path"]) if found is None or f.get("sha256") else None
            if ptb is not None:
                ptb_sha = hashlib.sha256(ptb.read_bytes()).hexdigest()
                rec_sha = f.get("sha256")
                same = rec_sha in (None, ptb_sha) or (found is not None and found.read_bytes().rstrip(b"\n") == ptb.read_bytes().rstrip(b"\n"))
                if found is None and rec_sha not in (None, ptb_sha):
                    same = False
                if same:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ptb, dest)
                    f["materialized"] = str(dest.relative_to(x_dir))
                    f["materialized_from"] = "third_party/PostTrainBench/src/eval/templates"
                    f["sha256"] = ptb_sha
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
        # Steps that run the scientist's own code must carry its launch-time content; copies,
        # checkpoint selection, base-model downloads and plain config edits need not.
        CODE_KINDS = {"train", "weight_average", "convert"}   # a data_build may be a shell one-liner
        entry_missing = [s["step_id"] for s in rec.get("steps", []) if s.get("kind") in CODE_KINDS
                         and not any(f.get("role") in ("entrypoint", "inline_script")
                                     and not f.get("source", "").startswith("unavailable")
                                     for f in s.get("files", []))]
        if entry_missing:
            errs.append("steps_without_available_entrypoint:" + ",".join(entry_missing))
        ver = bench / "x_verify" / cell / f"{cid}.json"
        verdict = json.loads(ver.read_text()).get("verdict") if ver.exists() else None
        rec["_verification"] = {"verdict": verdict, "path": str(ver.relative_to(bench)) if ver.exists() else None}
        (x_dir / "launch_record.json").write_text(json.dumps(rec, indent=1))
        records[cid], problems[cid] = rec, errs
    # splits: session level, with sessions that share any weights hash joined into one group
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    by_hash = {}
    locked = set()
    for row in labels:
        s = row["session_id"]
        find(s)
        if row.get("matrix_split") == "locked_session_test":
            locked.add(s)
        h = (row.get("weights") or {}).get("weights_sha256")
        if h:
            if h in by_hash and by_hash[h] != s:
                union(s, by_hash[h])
            by_hash.setdefault(h, s)
    members = defaultdict(set)
    for s in list(parent):
        members[find(s)].add(s)
    sessions, groups = {}, {}
    for root, mem in members.items():
        gid = "group-" + hashlib.sha256(("|".join(sorted(mem))).encode()).hexdigest()[:12]
        if mem & locked:
            split = "test"
        else:
            split = "validation" if int(hashlib.sha256((SALT + gid).encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
        for s in mem:
            sessions[s] = split
            groups[s] = gid
    multi = [sorted(m) for m in members.values() if len(m) > 1]
    # Weight identity: the matrix preflight hashed every archived checkpoint's shards. Two
    # checkpoint ids with the same weights_sha256 are the same weights (a card archived twice, or
    # two cards archiving one directory); under the same serving config the later one is an alias.
    weights_of = {}
    for row in labels:
        h = (row.get("weights") or {}).get("weights_sha256")
        if h:
            weights_of[row["checkpoint_id"]] = h
    # Alias = same weights + same S *within one session* (one directory archived under two cards).
    # The same weights produced independently in two sessions (same recipe, same seed) stay two
    # examples, as the spec says, but their sessions must fall in the same split.
    canonical = {}
    for row in sorted(labels, key=lambda r: r["checkpoint_id"]):
        h = weights_of.get(row["checkpoint_id"])
        if h:
            canonical.setdefault((row["session_id"], h, row["serving_id"], row["benchmark"]), row["checkpoint_id"])
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
        h = weights_of.get(cid)
        alias_of = None
        ckey = (row["session_id"], h, row["serving_id"], row["benchmark"])
        if h and canonical.get(ckey) not in (None, cid):
            alias_of = f"{canonical[ckey]}@{row['serving_id']}"
            reasons.append(f"alias:{alias_of}")
        zref = bench / "z" / row["example_id"] / "log_ref.json"
        z = json.loads(zref.read_text()) if zref.exists() else None
        if z is not None and not z.get("log_is_record_of_label"):
            reasons.append("z:log_not_record_of_label")
        ex = {
            "example_id": row["example_id"], "checkpoint_id": cid, "serving_id": row["serving_id"],
            "session_id": row["session_id"], "split": sessions[row["session_id"]], "split_group": groups[row["session_id"]],
            "benchmark": row["benchmark"], "base_model": row["base_model"], "track": row["track"],
            "status": "eligible" if not reasons else ("alias" if alias_of and all(r.startswith("alias:") for r in reasons) else "excluded"),
            "reasons": reasons, "alias_of": alias_of, "weights_sha256": h,
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
    (bench / "splits.json").write_text(json.dumps({"salt": SALT,
                                                   "rule": "sessions sharing a weights hash form one group; a group containing a matrix locked_session_test session -> test; "
                                                           "otherwise sha256(salt+group_id)%5==0 -> validation else train",
                                                   "multi_session_groups": multi, "sessions": sessions, "groups": groups}, indent=1))
    reason_counts = Counter("alias" if r.startswith("alias:") else (r.split(":")[0] + ":" + r.split(":")[1] if ":" in r else r)
                            for ex in examples for r in ex["reasons"])
    summary = {"examples": len(examples), "by_track_status": {f"{k[0]}/{k[1]}": v for k, v in sorted(status.items())},
               "checkpoints_with_record": len(records), "records_with_problems": sum(1 for v in problems.values() if v),
               "reasons": dict(reason_counts.most_common()),
               "eligible_by_split": dict(Counter(ex["split"] for ex in examples if ex["status"] == "eligible")),
               "eligible_by_benchmark": dict(Counter(ex["benchmark"] for ex in examples if ex["status"] == "eligible"))}
    (bench / "assemble_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv)
