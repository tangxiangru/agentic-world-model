"""Private offline archive scoring, gated on every frozen attempt being complete.

This module never launches or resumes an agent. The execution lock, byte-pinned
bundle, exact attempt inventory, capture hashes, launch metadata and normalized
results are checked BEFORE opening the label inventory. Completed failures stay
failures. Only the existing scorer joins selected recipes to official grades.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path

from tools.outcome_prediction import wm_evaluate as evaluate
from tools.outcome_prediction import wm_grade_inventory as grades
from tools.outcome_prediction import wm_run as runner

BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPLICATES = 5000


def _directory(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Expected a real, nonsymlink directory")
    return path.resolve(strict=True)


@contextmanager
def _execution_lock(directory):
    """Use the runner's existing lock; do not create or modify execution files."""
    descriptor = os.open(directory / ".lock", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Unsafe execution lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _exact_entries(directory, expected):
    if {p.name for p in directory.iterdir()} != set(expected):
        raise ValueError("Missing, extra or incomplete frozen attempt entries")


def verify_completed(bundle_path, *, bundle_sha256, execution_dir):
    """Read-only verifier; caller must hold the execution lock during label access.

    Returns the validated bundle, exact re-normalized runs, and raw-byte hashes.
    No label inventory is accepted or opened by this function.
    """
    bundle_path = Path(bundle_path).absolute()
    if bundle_path.is_symlink():
        raise ValueError("Bundle manifest must not be a symlink")
    bundle_path = bundle_path.resolve(strict=True)
    directory = _directory(execution_dir)
    bundle = runner.validate_bundle(bundle_path, expected_sha256=bundle_sha256)
    if bundle["protocol"]["label_source"] != "archived_checkpoint":
        raise ValueError("Archive scorer cannot score a fresh-execution protocol")
    identity = {
        "bundle_sha256": bundle_sha256,
        "protocol_sha256": evaluate.digest(bundle["protocol"]),
        "runner_sha256": runner.sha(Path(runner.__file__).read_bytes()),
    }
    study_raw = runner._read(directory / "study.json")
    if runner.strict_json(study_raw) != identity:
        raise ValueError("Execution belongs to a different frozen study")
    cases = evaluate.validate_protocol(bundle["protocol"])
    folders = {runner.sha(key.encode()): key for key in cases}
    if len(folders) != len(cases):
        raise ValueError("Ambiguous case directory identity")
    _exact_entries(directory, {".lock", "study.json", *folders})
    hashes, runs = {"study.json": runner.sha(study_raw)}, []
    for case in bundle["protocol"]["cases"]:
        case_id = case["case_id"]
        folder = _directory(directory / runner.sha(case_id.encode()))
        _exact_entries(folder, evaluate.ARMS)
        for arm in evaluate.ARMS:
            attempt = _directory(folder / arm)
            _exact_entries(attempt, runner.CAPTURE_FILES | {"result.json", "workspace"})
            _directory(attempt / "workspace")
            result_raw = runner._read(attempt / "result.json")
            result = runner.strict_json(result_raw)
            runner._keys(result, {"run", "capture_sha256"}, "attempt result")
            runner._keys(result["capture_sha256"], runner.CAPTURE_FILES, "capture fingerprints")
            prefix = str(attempt.relative_to(directory)) + "/"
            hashes[prefix + "result.json"] = runner.sha(result_raw)
            for name, expected in result["capture_sha256"].items():
                if not evaluate._hash(expected) or runner._file_sha(attempt / name) != expected:
                    raise ValueError("Capture SHA256 mismatch")
                hashes[prefix + name] = expected
            runner._verify_attempt_inputs(
                attempt, bundle, bundle_path.parent, case_id, arm, identity
            )
            capture = runner.strict_json(runner._read(attempt / "capture.json"))
            normalized = runner._normalized(attempt, bundle, case_id, arm, capture)
            if result["run"] != normalized:
                raise ValueError("Stored result differs from exact captured transcript")
            if normalized.get("case_id") != case_id or normalized.get("arm") != arm:
                raise ValueError("Normalized attempt identity mismatch")
            runs.append(normalized)
    if len(runs) != len(cases) * len(evaluate.ARMS):
        raise ValueError("Not every frozen attempt has completed")
    return bundle, runs, dict(sorted(hashes.items()))


def _archive_labels(inventory_path, inventory_sha256, protocol):
    """Private positive projection, called only after capture verification."""
    raw = runner._read(inventory_path, expected=inventory_sha256)
    cases = evaluate.validate_protocol(protocol)
    expected = {}
    for case in cases.values():
        for key in case["candidate_ids"]:
            cell = expected.setdefault(key, case["cell_id"])
            if cell != case["cell_id"]:
                raise ValueError("Candidate belongs to inconsistent frozen runs")
    values, seen = {}, set()
    projected_count = 0
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = grades.project(line)
        key, cell = row["example_id"], row["cell_id"]
        if (
            not isinstance(key, str)
            or not isinstance(cell, str)
            or not key.startswith(cell + "/")
            or key in seen
        ):
            raise ValueError("Invalid or duplicate archive identity")
        seen.add(key)
        projected_count += 1
        if key not in expected:
            continue
        if cell != expected[key] or not grades.valid_official_label(row["label"]):
            raise ValueError("Frozen candidate lacks a matching valid official final grade")
        values[key] = row["label"]["accuracy"]
    if set(values) != set(expected):
        raise ValueError("Missing frozen candidate final grades")
    return {
        case_id: {key: values[key] for key in case["candidate_ids"]}
        for case_id, case in cases.items()
    }, projected_count


def score_archive(
    bundle_path,
    *,
    bundle_sha256,
    execution_dir,
    inventory_path,
    inventory_sha256,
    output_dir,
):
    """Verify all attempts, then join labels and score once into a NEW directory."""
    if not evaluate._hash(bundle_sha256) or not evaluate._hash(inventory_sha256):
        raise ValueError("Explicit raw-byte SHA256 pins are required")
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError("Choose a NEW private scoring directory")
    execution = _directory(execution_dir)
    bundle_file = Path(bundle_path).absolute()
    inventory_file = Path(inventory_path).absolute()
    resolved_output = output.resolve()
    if any(
        resolved_output.is_relative_to(parent)
        for parent in (execution, bundle_file.parent.resolve(), inventory_file.parent.resolve())
    ):
        raise ValueError("Scoring output must be outside the immutable input directories")
    with _execution_lock(execution):
        bundle, runs, capture_hashes = verify_completed(
            bundle_file, bundle_sha256=bundle_sha256, execution_dir=execution
        )
        # No inventory file read, hashing, or JSON decoding occurs above this gate.
        labels, projected_count = _archive_labels(
            inventory_file, inventory_sha256, bundle["protocol"]
        )
        report = evaluate.score(
            bundle["protocol"],
            labels,
            runs,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
        )
        # Refuse to persist a result if capture bytes changed during the label join.
        for name, expected in capture_hashes.items():
            if runner._file_sha(execution / name) != expected:
                raise ValueError("Execution capture changed during offline scoring")
        runner._read(bundle_file, expected=bundle_sha256)
        outputs = {"labels.json": runner.encoded(labels), "report.json": runner.encoded(report)}
        provenance = {
            "schema": "wm-post-completion-archive-score-v1",
            "scope": "private_archived_checkpoint_scoring_not_fresh_execution",
            "bundle_sha256": bundle_sha256,
            "protocol_sha256": evaluate.digest(bundle["protocol"]),
            "inventory_sha256": inventory_sha256,
            "execution_files_sha256": capture_hashes,
            "source_files_sha256": bundle["runtime"]["source_files"],
            "scorer_sha256": runner.sha(Path(__file__).read_bytes()),
            "all_frozen_slots_completed_before_inventory_access": True,
            "attempts_re_normalized_from_preserved_captures": len(runs),
            "inference_launched": False,
            "retries_performed": False,
            "private_identity_and_final_label_objects_projected": projected_count,
            "plans_code_and_history_decoded_from_inventory": False,
            "labels_cover_exact_frozen_case_candidate_ids": True,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "files_sha256": {name: runner.sha(raw) for name, raw in outputs.items()},
            "limits": [
                "Completed failed decisions remain invalid; complete-pair estimates can be biased.",
                "Archive labels do not prove executed recipe or checkpoint fidelity.",
                "This is a trusted local harness, not an operating-system sandbox.",
            ],
        }
        output.mkdir(parents=True, mode=0o700)
        for name, raw in outputs.items():
            runner._write(output / name, raw, readonly=False)
        runner._write(output / "provenance.json", runner.encoded(provenance), readonly=False)
    return {
        "status": report["status"],
        "completed_attempts": len(runs),
        "planned_cases": len(bundle["protocol"]["cases"]),
        "paired_valid_cases": sum(row["paired_valid"] for row in report["cases"]),
        "labels_or_scores_in_summary": False,
        "output_dir": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bundle-path",
        "bundle-sha256",
        "execution-dir",
        "inventory-path",
        "inventory-sha256",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(score_archive(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
