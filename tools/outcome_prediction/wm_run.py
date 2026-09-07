"""Immutable input packaging and single-attempt execution of matched RPM/WM cases.

This is a trusted local harness, NOT an OS sandbox or semantic-review authority.
It accepts externally approved final-only candidate wrappers and packet audits;
it never opens a TEST-label file, chooses a cohort, trains, scores, or retries.

Source JSON keys: schema='wm-run-source-v1', protocol, split, runtime, model,
history, cases. split has train_cell_ids/test_cell_ids only. model has path and
sha256. history has root and manifest (wm_history). Each cases value has
candidate_payloads and packet_audit. Candidate wrappers have exactly example_id,
cell_id, model_input, content_review, step_sources. Use packet_fingerprints to
compute the protocol's hashes BEFORE obtaining external packet approval.

Hash meanings: protocol case evidence_sha256 is wm_evaluate.digest(files), where
files is the complete logical-path -> raw-byte-SHA256 allowlist. The candidate
and packet-audit hashes are wm_evaluate.digest(the exact JSON objects). CLI/Python,
model, source files, capture files and the top-level bundle hash use raw bytes.
The external packet audit must explicitly approve the exact three manifests and
attest outcome exclusion; hashes alone never manufacture those attestations.

package writes a NEW immutable bundle. execute requires --execute plus its exact
raw bundle hash. A locked output directory claims each case/arm before launch.
Finished attempts, including failures, are verified and reused, never rerun.
An interrupted claimed slot stops resumption for manual review, never retries.
Only the separate offline wm_evaluate scorer should later receive TEST labels.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import time
from pathlib import Path

from tools.outcome_prediction import wm_agent as agent
from tools.outcome_prediction import wm_evaluate as evaluate
from tools.outcome_prediction import wm_final_model as final_model
from tools.outcome_prediction import wm_history as history_tools
from tools.outcome_prediction.wm_corpus import SourceRoot
from tools.outcome_prediction.wm_evidence import _logical_path

SCHEMA = "wm-run-bundle-v1"
SOURCE_SCHEMA = "wm-run-source-v1"
ENV_KEYS = {"HOME", "PATH", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE"}
# Installed CLI 2.1.261's retry getter explicitly accepts zero. This also prevents
# hidden context compaction; startup/transcript checks remain independently required.
EXEC_ENV = {**agent.ENV_OVERRIDES, "CLAUDE_CODE_MAX_RETRIES": "0", "DISABLE_AUTO_COMPACT": "1"}
WRAPPER_KEYS = {"example_id", "cell_id", "model_input", "content_review", "step_sources"}
CAPTURE_FILES = {
    "transcript.jsonl",
    "stderr.log",
    "audit.jsonl",
    "server.json",
    "started.json",
    "capture.json",
}
CAPTURE_LIMITS = {
    "transcript.jsonl": 64 * 1024**2,
    "audit.jsonl": 16 * 1024**2,
    "stderr.log": 4 * 1024**2,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def strict_json(raw):
    return agent._load(raw.decode("utf-8"))


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Unexpected {name} fields")


def _read(path, expected=None):
    path = Path(path).absolute()
    with SourceRoot(path.parent) as root:
        return root.read(path.name, expected=expected)


def _write(path, raw, *, readonly=True):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o400 if readonly else 0o600)


def _file_sha(path):
    path = Path(path).absolute()
    with SourceRoot(path.parent) as root, os.fdopen(root._open(path.name), "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Capture must be a regular file")
        value = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024**2), b""):
            value.update(chunk)
        return value.hexdigest()


def _oversized(directory):
    return [
        name
        for name, limit in CAPTURE_LIMITS.items()
        if (directory / name).exists() and (directory / name).stat().st_size > limit
    ]


def execution_environment():
    """OAuth/keychain environment only; never return or persist credentials."""
    return {**{key: os.environ[key] for key in ENV_KEYS if key in os.environ}, **EXEC_ENV}


def runtime_spec(repository_root, python_executable, claude_executable):
    """Read-only metadata helper; no model/auth call and no approval creation."""
    repository = Path(repository_root).resolve(strict=True)
    paths = sorted((repository / "tools/outcome_prediction").glob("*.py"))
    return {
        "repository_root": str(repository),
        "python_executable": str(Path(python_executable).absolute()),
        "python_sha256": sha(_read(Path(python_executable).resolve(strict=True))),
        "claude_executable": str(Path(claude_executable).absolute()),
        "claude_sha256": sha(_read(Path(claude_executable).resolve(strict=True))),
        "source_files": {str(path.relative_to(repository)): sha(_read(path)) for path in paths},
    }


def auth_metadata(claude_executable):
    """Only version/login booleans; never expose account, key or token values."""
    env = execution_environment()
    version = subprocess.run(
        [str(claude_executable), "--version"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    status = subprocess.run(
        [str(claude_executable), "auth", "status", "--json"],
        env=env,
        capture_output=True,
        timeout=15,
        check=False,
    )
    try:
        parsed = strict_json(status.stdout)
    except (ValueError, UnicodeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    version_token = version.stdout.strip().split(" ")[0]
    return {
        "cli_version": version_token
        if version.returncode == 0 and re.fullmatch(r"\d+\.\d+\.\d+", version_token)
        else None,
        "oauth_logged_in": status.returncode == 0
        and parsed.get("loggedIn") is True
        and parsed.get("authMethod") in {"claude.ai", "oauth", "oauth_token"},
        "model_connectivity_tested": False,
    }


def packet_fingerprints(case_id, candidates, history_manifest):
    """Deterministic hashes only; does NOT create or approve a packet audit."""
    files = {
        "history/" + _logical_path(path): value for path, value in history_manifest["files"].items()
    }
    contents = {}
    for candidate_id, wrapper in candidates.items():
        payload = final_model.approved_payload(wrapper)
        path = f"candidates/{sha(case_id.encode())}/{sha(candidate_id.encode())}.json"
        raw = encoded({"candidate_id": candidate_id, "model_input": payload})
        files[path] = sha(raw)
        contents[path] = raw
    return {
        "files": files,
        "contents": contents,
        "evidence_sha256": evaluate.digest(files),
        "candidate_payloads_sha256": evaluate.digest(candidates),
        "history_manifest_sha256": evaluate.digest(history_manifest),
    }


def _validate(spec, *, bundle_root=None):
    _keys(spec, {"schema", "protocol", "split", "runtime", "model", "history", "cases"}, "bundle")
    if spec["schema"] != (SCHEMA if bundle_root else SOURCE_SCHEMA):
        raise ValueError("Unsupported bundle schema")
    protocol = spec["protocol"]
    cases = evaluate.validate_protocol(protocol)
    _keys(spec["split"], {"train_cell_ids", "test_cell_ids"}, "frozen split")
    train, test = (spec["split"][key] for key in ("train_cell_ids", "test_cell_ids"))
    if any(
        not isinstance(values, list)
        or not values
        or len(values) != len(set(values))
        or any(not isinstance(v, str) or not v for v in values)
        for values in (train, test)
    ):
        raise ValueError("Invalid frozen run partition")
    if set(train) & set(test) or {c["cell_id"] for c in cases.values()} - set(test):
        raise ValueError("Candidate/TRAIN overlap or candidate outside frozen TEST")
    runtime = spec["runtime"]
    _keys(
        runtime,
        {
            "repository_root",
            "python_executable",
            "python_sha256",
            "claude_executable",
            "claude_sha256",
            "source_files",
        },
        "runtime",
    )
    for name in ("repository_root", "python_executable", "claude_executable"):
        if not Path(runtime[name]).is_absolute():
            raise ValueError("Runtime paths must be absolute")
    for name in ("python", "claude"):
        _read(Path(runtime[f"{name}_executable"]).resolve(strict=True), runtime[f"{name}_sha256"])
    source_paths = {
        str(p.relative_to(runtime["repository_root"]))
        for p in (Path(runtime["repository_root"]) / "tools/outcome_prediction").glob("*.py")
    }
    if (
        set(runtime["source_files"]) != source_paths
        or "tools/outcome_prediction/wm_run.py" not in source_paths
    ):
        raise ValueError("Runtime source inventory changed")
    with SourceRoot(runtime["repository_root"]) as root:
        for path, expected in runtime["source_files"].items():
            root.read(path, expected=expected)
    if runtime["source_files"]["tools/outcome_prediction/wm_run.py"] != sha(
        Path(__file__).read_bytes()
    ):
        raise ValueError("Executing runner differs from pinned source")
    _keys(spec["model"], {"path", "sha256"}, "model")
    if spec["model"]["sha256"] != protocol["wm_model_sha256"]:
        raise ValueError("Model/protocol hash mismatch")
    model_path = (
        (bundle_root / _logical_path(spec["model"]["path"]))
        if bundle_root
        else Path(spec["model"]["path"])
    )
    if not model_path.is_absolute():
        raise ValueError("Source model path must be absolute")
    _read(model_path, spec["model"]["sha256"])
    # Trusted local pickle ONLY, hash-pinned before unpickling. load checks the
    # final-only model type and current feature contract; no prediction is called.
    model = final_model.FinalRecipeWorldModel.load(
        model_path, expected_sha256=spec["model"]["sha256"]
    )
    manifest = model.training_manifest
    fitted_cells = set(manifest["fitted_train_cell_ids"])
    empty_cells = set(manifest["empty_train_cell_ids"])
    if (
        set(manifest["train_cell_ids"]) != set(train)
        or set(manifest["forbidden_cell_ids"]) != set(test)
        or manifest.get("test_data_used") is not False
        or fitted_cells | empty_cells != set(train)
        or fitted_cells & empty_cells
        or {row["cell_id"] for row in manifest["training_examples"]} != fitted_cells
        or manifest["spec"] != final_model.SPEC
        or {case["benchmark"] for case in cases.values()} - set(model.models)
    ):
        raise ValueError("WM is not trained on the frozen TRAIN-only partition")
    _keys(spec["history"], {"root", "manifest"}, "history")
    historical = spec["history"]["manifest"]
    history_tools.validate_history_for_candidates(historical, candidate_run_ids=test)
    if set(historical["history_run_ids"]) != set(train) or set(
        historical["intended_candidate_run_ids"]
    ) != set(test):
        raise ValueError("History does not match frozen disjoint TRAIN runs")
    evidence_root = bundle_root / "evidence" if bundle_root else Path(spec["history"]["root"])
    if not evidence_root.is_absolute():
        raise ValueError("Source history root must be absolute")
    if bundle_root and spec["history"]["root"] != "evidence/history":
        raise ValueError("Unexpected packaged history root")
    with SourceRoot(evidence_root) as root:
        for path, expected in historical["files"].items():
            raw = root.read(("history/" if bundle_root else "") + path, expected=expected)
            raw.decode("utf-8")
    if set(spec["cases"]) != set(cases):
        raise ValueError("Bundle must contain every frozen case, no more or less")
    fingerprints = {}
    for case_id, case in cases.items():
        entry = spec["cases"][case_id]
        _keys(
            entry,
            {"candidate_payloads", "packet_audit"} | ({"files"} if bundle_root else set()),
            "case bundle",
        )
        candidates = entry["candidate_payloads"]
        if not isinstance(candidates, dict) or set(candidates) != set(case["candidate_ids"]):
            raise ValueError("Candidate IDs differ from frozen case")
        for candidate_id, wrapper in candidates.items():
            _keys(wrapper, WRAPPER_KEYS, "candidate wrapper (labels forbidden)")
            if wrapper["example_id"] != candidate_id:
                raise ValueError("Candidate key must equal its archive target example_id")
            payload = final_model.approved_payload(wrapper)
            sources = wrapper["step_sources"]
            if (
                wrapper["cell_id"] != case["cell_id"]
                or payload["task"]["benchmark"] != case["benchmark"]
                or set(sources) != {s["step_id"] for s in payload["recipe"]["steps"]}
                or sources.get(payload["recipe"]["final_step_id"]) != wrapper["example_id"]
                or any(
                    not isinstance(s, str) or not s.startswith(case["cell_id"] + "/")
                    for s in sources.values()
                )
            ):
                raise ValueError("Candidate task/source identity mismatch")
        fp = packet_fingerprints(case_id, candidates, historical)
        if any(fp[key] != case[key] for key in ("evidence_sha256", "candidate_payloads_sha256")):
            raise ValueError("Frozen candidate/evidence hash mismatch")
        audit = entry["packet_audit"]
        required = {
            "schema": "wm-case-packet-audit-v1",
            "case_id": case_id,
            "status": "approved",
            "candidate_payloads_sha256": fp["candidate_payloads_sha256"],
            "evidence_sha256": fp["evidence_sha256"],
            "history_manifest_sha256": fp["history_manifest_sha256"],
            "candidate_ancestor_outcomes_excluded": True,
            "test_outcomes_excluded": True,
            "same_evidence_both_arms": True,
        }
        _keys(audit, set(required) | {"reviewer", "review_evidence_sha256"}, "packet audit")
        if (
            any(audit[k] != v for k, v in required.items())
            or not isinstance(audit["reviewer"], str)
            or not audit["reviewer"].strip()
            or not evaluate._hash(audit["review_evidence_sha256"])
            or evaluate.digest(audit) != case["packet_audit_sha256"]
        ):
            raise ValueError("Missing, stale or unapproved packet exposure audit")
        if bundle_root:
            if entry["files"] != fp["files"]:
                raise ValueError("Shared evidence allowlist differs from frozen case")
            with SourceRoot(evidence_root) as root:
                for path, raw in fp["contents"].items():
                    if root.read(path, expected=fp["files"][path]) != raw:
                        raise ValueError("Candidate evidence bytes differ from approved payload")
        # Also validate the exact common prompts/budgets before any CLI invocation.
        agent.build_command(
            protocol,
            case_id,
            "rpm",
            python_executable=runtime["python_executable"],
            repository_root=runtime["repository_root"],
            server_config_path="/unused/config.json",
            claude_executable=runtime["claude_executable"],
        )
        fingerprints[case_id] = fp
    return fingerprints


def package(source, output_dir):
    """Validate externally approved material and copy only exact allowlisted bytes."""
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError("Choose a NEW immutable bundle directory")
    fingerprints = _validate(source)
    output.mkdir(parents=True, mode=0o700)
    bundle = copy.deepcopy(source)
    bundle["schema"] = SCHEMA
    bundle["model"]["path"] = "model.joblib"
    bundle["history"]["root"] = "evidence/history"
    _write(output / "model.joblib", _read(source["model"]["path"], source["model"]["sha256"]))
    with SourceRoot(source["history"]["root"]) as root:
        for path, expected in source["history"]["manifest"]["files"].items():
            _write(output / "evidence/history" / path, root.read(path, expected=expected))
    for case_id, fp in fingerprints.items():
        bundle["cases"][case_id]["files"] = fp["files"]
        for path, raw in fp["contents"].items():
            _write(output / "evidence" / path, raw)
    raw = encoded(bundle)
    _write(output / "bundle.json", raw)  # completion marker written LAST
    return {"bundle_path": str(output / "bundle.json"), "bundle_sha256": sha(raw)}


def validate_bundle(bundle_path, *, expected_sha256):
    path = Path(bundle_path).absolute()
    raw = _read(path, expected_sha256)
    bundle = strict_json(raw)
    _validate(bundle, bundle_root=path.parent)
    return bundle


def _server_config(bundle, bundle_root, case_id, arm, audit_path):
    entry = bundle["cases"][case_id]
    config = {
        "evidence_root": str(bundle_root / "evidence"),
        "files": entry["files"],
        "audit_log": str(audit_path),
    }
    if arm == "rpm_wm":
        config.update(
            model_path=str(bundle_root / bundle["model"]["path"]),
            model_sha256=bundle["model"]["sha256"],
            candidate_payloads=entry["candidate_payloads"],
        )
    return config


def _terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


def _launch(spec, directory):
    """Exactly one process; all output stays in private capture files."""
    started = time.monotonic()
    timed_out, failure, returncode = False, None, None
    process = None
    with (
        (directory / "transcript.jsonl").open("xb") as stdout,
        (directory / "stderr.log").open("xb") as stderr,
    ):
        try:
            process = subprocess.Popen(
                spec["argv"],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                cwd=directory / "workspace",
                env=execution_environment(),
                start_new_session=True,
            )
            data = spec["stdin"].encode()
            while True:
                remaining = spec["timeout_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("pinned_cli", spec["timeout_seconds"])
                try:
                    process.communicate(data, timeout=min(1, remaining))
                    returncode = process.returncode
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= spec["timeout_seconds"]:
                        raise
                    if _oversized(directory):
                        failure = "capture_size_limit"
                        _terminate(process)
                        returncode = process.returncode
                        break
                    data = None
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process)
            returncode = process.returncode
        except (OSError, KeyboardInterrupt) as error:
            failure = type(error).__name__
            if process is not None:
                _terminate(process)
                returncode = process.returncode
    return {
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "timed_out": timed_out,
        "runner_failure": failure,
    }


def _normalized(directory, bundle, case_id, arm, capture):
    _keys(capture, {"returncode", "elapsed_seconds", "timed_out", "runner_failure"}, "capture")
    oversized = _oversized(directory)
    raw = b"" if "transcript.jsonl" in oversized else _read(directory / "transcript.jsonl")
    audit = b"" if "audit.jsonl" in oversized else _read(directory / "audit.jsonl")
    run = agent.normalize_stream(
        raw,
        audit,
        protocol=bundle["protocol"],
        case_id=case_id,
        arm=arm,
        returncode=capture["returncode"],
        elapsed_seconds=capture["elapsed_seconds"],
        timed_out=capture["timed_out"],
    )
    if capture["runner_failure"]:
        run["status"] = "invalid"
        run["normalization"]["failure_reason"] = "runner_" + capture["runner_failure"]
    if oversized:
        run["status"] = "invalid"
        run["normalization"]["failure_reason"] = "capture_size_limit"
        run["transcript_sha256"] = _file_sha(directory / "transcript.jsonl")
        run["server_audit_sha256"] = _file_sha(directory / "audit.jsonl")
    return run


def _verify_attempt_inputs(directory, bundle, bundle_root, case_id, arm, identity):
    expected = _server_config(bundle, bundle_root, case_id, arm, directory / "audit.jsonl")
    if strict_json(_read(directory / "server.json")) != expected:
        raise ValueError("Attempt served different evidence/model/candidates")
    runtime = bundle["runtime"]
    command = agent.build_command(
        bundle["protocol"],
        case_id,
        arm,
        python_executable=runtime["python_executable"],
        repository_root=runtime["repository_root"],
        server_config_path=str(directory / "server.json"),
        claude_executable=runtime["claude_executable"],
    )
    started = strict_json(_read(directory / "started.json"))
    if (
        started["identity"] != identity
        or started["case_id"] != case_id
        or started["arm"] != arm
        or started["command"] != command
        or started["additional_environment_overrides"]
        != {k: EXEC_ENV[k] for k in ("CLAUDE_CODE_MAX_RETRIES", "DISABLE_AUTO_COMPACT")}
    ):
        raise ValueError("Attempt launch differs from frozen runtime")


def execute(bundle_path, *, expected_sha256, output_dir, authorized=False):
    """Run every frozen slot once, or resume only untouched slots. No scoring."""
    if authorized is not True:
        raise PermissionError("Explicit --execute authorization is required")
    bundle_path = Path(bundle_path).absolute()
    if bundle_path.is_symlink():
        raise ValueError("Bundle manifest must not be a symlink")
    bundle_path = bundle_path.resolve(strict=True)
    bundle = validate_bundle(bundle_path, expected_sha256=expected_sha256)
    original_output = Path(output_dir).absolute()
    output = original_output.resolve()
    if original_output.is_symlink() or output.is_relative_to(bundle_path.parent):
        raise ValueError("Capture output must be outside the immutable bundle")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_fd = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        lock_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise ValueError("Unsafe study lock")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "bundle_sha256": expected_sha256,
            "protocol_sha256": evaluate.digest(bundle["protocol"]),
            "runner_sha256": sha(Path(__file__).read_bytes()),
        }
        if (output / "study.json").exists():
            if strict_json(_read(output / "study.json")) != identity:
                raise ValueError("Capture directory belongs to another frozen study")
        else:
            if {p.name for p in output.iterdir()} != {".lock"}:
                raise ValueError("Unrecognized partial output; manual review required")
            _write(output / "study.json", encoded(identity))
        slots = [
            (case["case_id"], arm) for case in bundle["protocol"]["cases"] for arm in evaluate.ARMS
        ]
        # Precheck ALL claimed slots before launching any new work.
        runs, pending = [], []
        for case_id, arm in slots:
            directory = output / sha(case_id.encode()) / arm
            if directory.parent.is_symlink():
                raise ValueError("Attempt directories must not traverse symlinks")
            if not directory.exists():
                pending.append((case_id, arm, directory))
                continue
            if directory.is_symlink() or not (directory / "result.json").is_file():
                raise ValueError(
                    "Interrupted claimed attempt; manual review required, no automatic retry"
                )
            result = strict_json(_read(directory / "result.json"))
            _keys(result, {"run", "capture_sha256"}, "attempt result")
            _keys(result["capture_sha256"], CAPTURE_FILES, "capture fingerprints")
            for name, expected in result["capture_sha256"].items():
                if _file_sha(directory / name) != expected:
                    raise ValueError("Capture SHA256 mismatch")
            _verify_attempt_inputs(directory, bundle, bundle_path.parent, case_id, arm, identity)
            capture = strict_json(_read(directory / "capture.json"))
            normalized = _normalized(directory, bundle, case_id, arm, capture)
            if result["run"] != normalized:
                raise ValueError("Stored result differs from exact captured transcript")
            runs.append(normalized)
        for case_id, arm, directory in pending:
            # Recheck all pinned inputs immediately before every invocation.
            bundle = validate_bundle(bundle_path, expected_sha256=expected_sha256)
            metadata = auth_metadata(bundle["runtime"]["claude_executable"])
            if (
                metadata["cli_version"] != bundle["protocol"]["agent"]["cli_version"]
                or not metadata["oauth_logged_in"]
            ):
                raise ValueError(
                    "Pinned CLI version or OAuth metadata unavailable; no inference launched"
                )
            directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.mkdir(mode=0o700)
            (directory / "workspace").mkdir(mode=0o700)
            config = _server_config(
                bundle, bundle_path.parent, case_id, arm, directory / "audit.jsonl"
            )
            _write(directory / "server.json", encoded(config))
            _write(directory / "audit.jsonl", b"", readonly=False)
            runtime = bundle["runtime"]
            spec = agent.build_command(
                bundle["protocol"],
                case_id,
                arm,
                python_executable=runtime["python_executable"],
                repository_root=runtime["repository_root"],
                server_config_path=str(directory / "server.json"),
                claude_executable=runtime["claude_executable"],
            )
            _write(
                directory / "started.json",
                encoded(
                    {
                        "identity": identity,
                        "case_id": case_id,
                        "arm": arm,
                        "command": spec,
                        "auth_metadata": metadata,
                        "additional_environment_overrides": {
                            k: EXEC_ENV[k]
                            for k in ("CLAUDE_CODE_MAX_RETRIES", "DISABLE_AUTO_COMPACT")
                        },
                    }
                ),
            )
            capture = _launch(spec, directory)
            _write(directory / "capture.json", encoded(capture))
            _verify_attempt_inputs(directory, bundle, bundle_path.parent, case_id, arm, identity)
            run = _normalized(directory, bundle, case_id, arm, capture)
            hashes = {name: _file_sha(directory / name) for name in sorted(CAPTURE_FILES)}
            _write(directory / "result.json", encoded({"run": run, "capture_sha256": hashes}))
            runs.append(run)
            if capture["runner_failure"] == "KeyboardInterrupt":
                break
        return runs
    finally:
        os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--bundle-sha256", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "package":
        print(json.dumps(package(strict_json(_read(args.source)), args.output_dir)))
    else:
        runs = execute(
            args.bundle,
            expected_sha256=args.bundle_sha256,
            output_dir=args.output_dir,
            authorized=args.execute,
        )
        print(
            json.dumps(
                {
                    "completed_slots": len(runs),
                    "valid_decisions": sum(r["status"] == "success" for r in runs),
                    "scored": False,
                }
            )
        )


if __name__ == "__main__":
    main()
