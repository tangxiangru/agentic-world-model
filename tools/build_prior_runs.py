"""Materialise the prior-runs directory a scientist gets to read.

For every run of a split (train side, or train + test), copy exactly the
intended corpus files into ``<out>/<agent config>/<run>/`` and write
``INDEX.md`` / ``index.jsonl`` with base model, agent, official accuracy, wall
time, and path. ``corpus-manifest.json`` pins the declared split, dataset
revision, visible sides, and SHA-256/size of every exposed per-run file. The
bundle contains the complete ``solve_out.txt`` session trajectory plus
``metrics.json`` and ``time_taken.txt``. It deliberately excludes upstream
``task/`` workspace snapshots and every optional, unattested run artifact.
Nothing is masked: this is the full-information baseline, and the study
decision (2026-08-31) is that scores and agent identity stay visible.

Two versions per split, chosen with ``--sides``:

    python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_193 --sides train,test
    python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_143 --sides train

Copy mode refuses an existing output by default. Pass ``--replace`` to stage a
complete replacement and swap it into place only after every declared run has
validated against the pinned Hugging Face download metadata. ``--index-only``
never copies, replaces, or re-attests data: it verifies that an existing output
contains exactly the declared runs and matches its immutable manifest before
rebuilding indexes from those copies.

The output is meant to be bind-mounted read-only at ``/home/ben/prior_runs``
(see rollout/patches/apply_extra_binds.py); it is a copy, not symlinks, so it
resolves inside the sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_RE = re.compile(r"^(?P<bench>[^_]+)_(?P<model>.+)_(?P<cid>\d+)$")
MANDATORY_FILES = ("solve_out.txt", "metrics.json", "time_taken.txt")
MANIFEST_FILE = "corpus-manifest.json"
MANIFEST_SCHEMA = "awm-prior-runs-v1"
METADATA_FILES = frozenset(("INDEX.md", "index.jsonl", "README.md", MANIFEST_FILE))


class PriorRunsError(RuntimeError):
    """The requested corpus cannot be materialised without leaking or omitting data."""


def _normalise_runs(runs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not runs:
        raise PriorRunsError("no runs were declared")
    normalised: list[tuple[str, str]] = []
    seen: set[str] = set()
    for run, side in runs:
        posix = PurePosixPath(run)
        if (
            posix.is_absolute()
            or len(posix.parts) != 2
            or any(part in ("", ".", "..") for part in posix.parts)
            or "\\" in run
        ):
            raise PriorRunsError(f"invalid run path {run!r}; expected <agent-config>/<run-name>")
        config, run_name = posix.parts
        if config in METADATA_FILES or config.startswith(".") or not RUN_RE.fullmatch(run_name):
            raise PriorRunsError(f"invalid PostTrainBench run path: {run!r}")
        if side not in ("train", "test"):
            raise PriorRunsError(f"invalid side {side!r} for {run}; expected train or test")
        canonical = f"{config}/{run_name}"
        if run != canonical:
            raise PriorRunsError(f"run path is not canonical: {run!r} (expected {canonical!r})")
        if canonical in seen:
            raise PriorRunsError(f"run declared more than once: {canonical}")
        seen.add(canonical)
        normalised.append((canonical, side))
    return normalised


def _normalise_provenance(
    runs: list[tuple[str, str]],
    *,
    split_id: str,
    dataset: dict,
    sides: tuple[str, ...],
) -> dict:
    if not split_id or not isinstance(split_id, str):
        raise PriorRunsError("split_id is required")
    if (
        not sides
        or len(sides) != len(set(sides))
        or any(side not in ("train", "test") for side in sides)
    ):
        raise PriorRunsError("sides must contain train and/or test exactly once")
    declared_sides = {side for _run, side in runs}
    if declared_sides != set(sides):
        raise PriorRunsError(
            f"declared run sides {sorted(declared_sides)} do not match "
            f"provenance sides {list(sides)}"
        )
    required_dataset = ("repo", "repo_type", "revision")
    if not isinstance(dataset, dict) or any(not dataset.get(key) for key in required_dataset):
        raise PriorRunsError(f"dataset must define {', '.join(required_dataset)}")
    revision = dataset["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise PriorRunsError("dataset revision must be a full 40-hex commit")
    # Preserve the complete committed dataset contract (catalogue/hash included)
    # while detaching it from any mutable caller-owned mapping.
    try:
        frozen_dataset = json.loads(json.dumps(dataset, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise PriorRunsError(f"dataset provenance is not JSON-serializable: {exc}") from exc
    return {
        "split_id": split_id,
        "sides": list(sides),
        "dataset": frozen_dataset,
    }


def _validated_run_files(run: str, src: Path, *, copied: bool) -> tuple[dict, str]:
    problems: list[str] = []
    if not src.is_dir() or src.is_symlink():
        raise PriorRunsError(f"declared run directory is missing, linked, or invalid: {run} ({src})")
    for name in MANDATORY_FILES:
        path = src / name
        if not path.is_file() or path.is_symlink():
            problems.append(f"{run}/{name} is missing, linked, or not a regular file")
        elif path.stat().st_size == 0:
            problems.append(f"{run}/{name} is empty")
    if copied:
        for path in src.iterdir():
            if path.name not in MANDATORY_FILES or not path.is_file() or path.is_symlink():
                problems.append(
                    f"{run} contains an unexpected or unattested path: {path.name}"
                )
    if problems:
        raise PriorRunsError("; ".join(problems))

    metrics_path = src / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PriorRunsError(f"invalid {run}/metrics.json: {exc}") from exc
    if not isinstance(metrics, dict):
        raise PriorRunsError(f"invalid {run}/metrics.json: top level must be an object")
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        raise PriorRunsError(f"invalid {run}/metrics.json: numeric accuracy is required")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
    ):
        raise PriorRunsError(f"invalid {run}/metrics.json accuracy: {accuracy!r}")
    try:
        time_taken = (src / "time_taken.txt").read_text().strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise PriorRunsError(f"invalid {run}/time_taken.txt: {exc}") from exc
    if not time_taken:
        raise PriorRunsError(f"invalid {run}/time_taken.txt: value is blank")
    return metrics, time_taken


def run_record(run: str, side: str, src: Path) -> dict:
    config, run_name = run.split("/", 1)
    m = RUN_RE.match(run_name)
    model = m.group("model").replace("_", "/", 1) if m else None
    metrics, tt = _validated_run_files(run, src, copied=False)
    acc = metrics.get("accuracy")
    return {
        "run": run, "agent_config": config, "run_name": run_name, "side": side,
        "base_model": model, "accuracy": acc, "time_taken": tt,
        "has_trace": True,
        "trace_bytes": (src / "solve_out.txt").stat().st_size,
    }


def _validate_exact_output(
    runs: list[tuple[str, str]], out: Path, *, require_metadata: bool
) -> None:
    """Reject missing, stale, undeclared, or linked content in a materialised corpus."""
    if not out.is_dir() or out.is_symlink():
        raise PriorRunsError(f"prior-runs output is missing or not a real directory: {out}")

    expected_by_config: dict[str, set[str]] = defaultdict(set)
    for run, _side in runs:
        config, run_name = run.split("/", 1)
        expected_by_config[config].add(run_name)

    problems: list[str] = []
    for entry in out.iterdir():
        if entry.name in METADATA_FILES:
            if entry.exists() and (not entry.is_file() or entry.is_symlink()):
                problems.append(f"metadata path is not a regular copied file: {entry.name}")
        elif entry.name not in expected_by_config:
            problems.append(f"undeclared top-level content: {entry.name}")
        elif not entry.is_dir() or entry.is_symlink():
            problems.append(f"agent-config path is not a real directory: {entry.name}")

    if require_metadata:
        for name in METADATA_FILES:
            path = out / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                problems.append(f"required metadata file is missing or invalid: {name}")

    for config, expected_names in expected_by_config.items():
        config_dir = out / config
        if not config_dir.is_dir() or config_dir.is_symlink():
            problems.append(f"declared agent-config directory is missing or invalid: {config}")
            continue
        actual_names: set[str] = set()
        for child in config_dir.iterdir():
            if not child.is_dir() or child.is_symlink():
                problems.append(f"undeclared non-run content under {config}: {child.name}")
            else:
                actual_names.add(child.name)
        for missing in sorted(expected_names - actual_names):
            problems.append(f"declared run is missing from output: {config}/{missing}")
        for extra in sorted(actual_names - expected_names):
            problems.append(f"undeclared run is present in output: {config}/{extra}")
        for run_name in sorted(expected_names & actual_names):
            try:
                _validated_run_files(f"{config}/{run_name}", config_dir / run_name, copied=True)
            except PriorRunsError as exc:
                problems.append(str(exc))
    if problems:
        raise PriorRunsError("; ".join(problems))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_manifest(
    runs: list[tuple[str, str]], data_root: Path, provenance: dict
) -> dict:
    entries = []
    for run, side in runs:
        files = {}
        for name in MANDATORY_FILES:
            path = data_root / run / name
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        entries.append({"run": run, "side": side, "files": files})
    return {
        "schema_version": MANIFEST_SCHEMA,
        "split": {"id": provenance["split_id"], "sides": provenance["sides"]},
        "dataset": provenance["dataset"],
        "file_scope": list(MANDATORY_FILES),
        "run_count": len(entries),
        "runs": entries,
    }


def _write_manifest(runs: list[tuple[str, str]], data_root: Path, provenance: dict) -> dict:
    manifest = _corpus_manifest(runs, data_root, provenance)
    _atomic_write(
        data_root / MANIFEST_FILE,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _validate_manifest(runs: list[tuple[str, str]], data_root: Path, provenance: dict) -> dict:
    path = data_root / MANIFEST_FILE
    if not path.is_file() or path.is_symlink():
        raise PriorRunsError(f"required immutable manifest is missing or invalid: {MANIFEST_FILE}")
    try:
        manifest_text = path.read_text()
        manifest = json.loads(manifest_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PriorRunsError(f"invalid {MANIFEST_FILE}: {exc}") from exc
    expected = _corpus_manifest(runs, data_root, provenance)
    expected_text = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if manifest != expected or manifest_text != expected_text:
        raise PriorRunsError(
            f"{MANIFEST_FILE} does not match the declared split/revision/sides "
            "or current file hashes"
        )
    return manifest


def _verify_source_revision(runs: list[tuple[str, str]], raw_dir: Path, revision: str) -> None:
    from awm.traj.fetch import check_ptb_run_files

    problems = check_ptb_run_files(
        [run for run, _side in runs],
        revision=revision,
        dest=raw_dir,
        required_files=MANDATORY_FILES,
    )
    if problems:
        raise PriorRunsError("raw source provenance failed: " + "; ".join(problems))


def _write_metadata(runs: list[tuple[str, str]], data_root: Path, out: Path) -> dict:
    rows = []
    for run, side in runs:
        rec = run_record(run, side, data_root / run)
        rec["path"] = f"/home/ben/prior_runs/{run}"
        rows.append(rec)
    rows.sort(key=lambda r: (-(r["accuracy"] if r["accuracy"] is not None else -1), r["run"]))
    index_jsonl = "".join(json.dumps(row) + "\n" for row in rows)
    _atomic_write(data_root / "index.jsonl", index_jsonl)
    lines = [
        "# Prior runs",
        "",
        (f"{len(rows)} previous attempts at this task by autonomous agents, one directory each, "
         "laid out as `<agent config>/<run>/`. Each holds `solve_out.txt` (the agent's complete "
         "session trace), `metrics.json` (official accuracy), and `time_taken.txt`. "
         "No optional run artifacts or `task/` workspace snapshots are exposed."),
        "",
        "Sorted by official accuracy, best first.",
        "",
        "| accuracy | base model | agent config | time | trace | path |",
        "|---:|---|---|---|---:|---|",
    ]
    for r in rows:
        acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "—"
        lines.append(f"| {acc} | {r['base_model']} | {r['agent_config']} | {r['time_taken'] or '—'} | "
                     f"{r['trace_bytes'] // 1024} KB | `{r['path']}` |")
    _atomic_write(data_root / "INDEX.md", "\n".join(lines) + "\n")
    summary = {
        "runs": len(rows),
        "missing": [],
        "out": str(out),
        "manifest_sha256": _sha256(data_root / MANIFEST_FILE),
        "by_model": {},
        "by_side": {},
    }
    for r in rows:
        summary["by_model"][r["base_model"]] = summary["by_model"].get(r["base_model"], 0) + 1
        summary["by_side"][r["side"]] = summary["by_side"].get(r["side"], 0) + 1
    _atomic_write(
        data_root / "README.md",
        "Read-only copy of prior PostTrainBench runs for this task, built by "
        "tools/build_prior_runs.py. Start with INDEX.md.\n",
    )
    return summary


def _publish_staged(stage: Path, out: Path, *, replace: bool) -> None:
    if not out.exists():
        os.replace(stage, out)
        return
    if not replace:
        raise FileExistsError(f"output already exists; pass --replace to rebuild it: {out}")
    if not out.is_dir() or out.is_symlink():
        raise PriorRunsError(f"refusing to replace a non-directory or symlink: {out}")

    backup = out.with_name(f".{out.name}.old-{uuid.uuid4().hex}")
    out.rename(backup)
    try:
        os.replace(stage, out)
    except BaseException:
        backup.rename(out)
        raise
    shutil.rmtree(backup)


def build(
    runs: list[tuple[str, str]],
    raw_dir: Path,
    out: Path,
    *,
    split_id: str,
    dataset: dict,
    sides: tuple[str, ...],
    copy: bool = True,
    replace: bool = False,
) -> dict:
    """Materialise or validate exactly ``runs = [(run, side)]``."""
    declared = _normalise_runs(runs)
    provenance = _normalise_provenance(
        declared, split_id=split_id, dataset=dataset, sides=sides
    )
    raw_dir = Path(raw_dir)
    out = Path(os.path.abspath(out))
    if out == out.parent:
        raise PriorRunsError("refusing to use a filesystem root as --out")
    if out.is_symlink():
        raise PriorRunsError(f"refusing symlink output: {out}")

    if not copy:
        if replace:
            raise PriorRunsError("--replace cannot be combined with --index-only")
        _validate_exact_output(declared, out, require_metadata=False)
        _validate_manifest(declared, out, provenance)
        summary = _write_metadata(declared, out, out)
        _validate_exact_output(declared, out, require_metadata=True)
        _validate_manifest(declared, out, provenance)
        return summary

    if out.exists() and not replace:
        raise FileExistsError(f"output already exists; pass --replace to rebuild it: {out}")
    if out.exists() and (not out.is_dir() or out.is_symlink()):
        raise PriorRunsError(f"refusing to replace a non-directory or symlink: {out}")
    for run, _side in declared:
        _validated_run_files(run, raw_dir / run, copied=False)
    _verify_source_revision(declared, raw_dir, provenance["dataset"]["revision"])

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        for run, _side in declared:
            src = raw_dir / run
            dst = stage / run
            dst.mkdir(parents=True)
            for name in MANDATORY_FILES:
                shutil.copy2(src / name, dst / name)
        # A resumed local Hub directory can be mutated concurrently. Recheck
        # its sidecars after copying before attesting to the staged bytes.
        _verify_source_revision(declared, raw_dir, provenance["dataset"]["revision"])
        _validate_exact_output(declared, stage, require_metadata=False)
        _write_manifest(declared, stage, provenance)
        summary = _write_metadata(declared, stage, out)
        _validate_exact_output(declared, stage, require_metadata=True)
        _validate_manifest(declared, stage, provenance)
        _publish_staged(stage, out, replace=replace)
        return summary
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("split_id", help="e.g. posttrainbench/gsm8k-gemma-holdout-v1")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sides", default="train", help="comma list of train,test (default train)")
    ap.add_argument("--raw-dir", type=Path, help="override <data>/traj/raw/posttrainbench")
    ap.add_argument(
        "--index-only",
        action="store_true",
        help="validate exact existing contents and rebuild metadata without copying",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing copy-mode output after staging a complete validated rebuild",
    )
    a = ap.parse_args()

    from awm import paths, splits

    s = splits.load(a.split_id)
    sides = [x.strip() for x in a.sides.split(",") if x.strip()]
    if (
        not sides
        or len(sides) != len(set(sides))
        or any(side not in ("train", "test") for side in sides)
    ):
        ap.error("--sides must be a non-empty comma list containing train and/or test once each")
    if a.index_only and a.replace:
        ap.error("--replace cannot be combined with --index-only")
    runs = [(r, side) for side in sides for r in getattr(s, side)]
    raw = a.raw_dir or paths.raw_dir("posttrainbench")
    try:
        summary = build(
            runs,
            raw,
            a.out,
            split_id=a.split_id,
            dataset=s.dataset,
            sides=tuple(sides),
            copy=not a.index_only,
            replace=a.replace,
        )
    except (PriorRunsError, FileExistsError) as exc:
        ap.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
