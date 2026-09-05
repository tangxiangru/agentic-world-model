"""Explicit environment operations and frozen placement, separate from science."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace

KIND = "environment-acceptance"
REPORT_SCHEMA = "ptb-humaneval-environment-v1"
SOURCE_PATHS = {
    "probe": "src/eval/humaneval_environment_probe.py",
    "runner": "src/commit_utils/slurm/humaneval_environment_acceptance.py",
    "helper": "src/eval/ptb_python_sandbox.py",
}
DATASET = {"sha256": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
           "bytes": 83920, "rows": 164}
SOURCE_RUNTIME_SHA256 = {
    "opus_5.sif": "8f0ce4b3c4702f2d8b497cd35dd6f42322c71af04a6fa906ea423ad790cc515c",
    "vllm_debug.sif": "2d7606e28dfd14ed33e5940ce62ca0961319239a85ac85cde1147e57b65be2a7",
}


class EnvironmentError(ValueError):
    pass


def is_operation(document: dict) -> bool:
    return isinstance(document.get("operation"), dict) and document["operation"].get("kind") == KIND


def validate_operation(document: dict) -> None:
    if "operation" not in document:
        return
    operation = document["operation"]
    if not isinstance(operation, dict) or set(operation) != {
        "kind", "target", "walltime", "probe_sha256", "runner_sha256", "helper_sha256"
    }:
        raise EnvironmentError("invalid environment operation fields")
    if (operation["kind"], operation["target"], operation["walltime"]) != (
        KIND, "humaneval", "00:15:00"
    ):
        raise EnvironmentError("only bounded HumanEval environment acceptance is supported")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(operation[f"{name}_sha256"])) for name in SOURCE_PATHS):
        raise EnvironmentError("environment operation must freeze all probe source hashes")
    if document.get("pilot") or document.get("contract", {}).get("task") != "gsm8k":
        raise EnvironmentError("environment operation requires a GSM8K reference profile and no pilot")
    if any(cell.get("awm") for cell in document.get("cells", [])):
        raise EnvironmentError("environment acceptance must not install a scientist treatment")
    if "placement" not in document:
        raise EnvironmentError("environment acceptance requires explicit node placement")


def requested_nodes(document: dict) -> set[str] | None:
    if "placement" not in document:
        return None
    placement = document["placement"]
    if not isinstance(placement, dict) or set(placement) != {"requested_nodes"}:
        raise EnvironmentError("placement must contain only requested_nodes")
    nodes = placement["requested_nodes"]
    if (not isinstance(nodes, list) or not nodes
            or any(not isinstance(node, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", node) for node in nodes)
            or len(set(nodes)) != len(nodes)):
        raise EnvironmentError("requested_nodes must be distinct, fully expanded node names")
    return set(nodes)


def validate_admission_reference(document: dict) -> dict:
    reference = document.get("environment_admission")
    if (not isinstance(reference, dict) or set(reference) != {"receipt", "sha256"}
            or not isinstance(reference.get("receipt"), str)
            or not reference["receipt"].startswith("results/ptb/")
            or not reference["receipt"].endswith(".json")
            or ".." in Path(reference["receipt"]).parts
            or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", "")))):
        raise EnvironmentError("HumanEval requires a frozen receipt-backed environment admission")
    if requested_nodes(document) is None:
        raise EnvironmentError("HumanEval requires explicitly admitted requested_nodes")
    return reference


def effective_nodes(document: dict, site_nodes: set[str]) -> set[str]:
    requested = requested_nodes(document)
    if requested is None:
        return set(site_nodes)
    if not requested <= site_nodes:
        raise EnvironmentError("requested placement is outside frozen site nodes")
    return requested


def environment_output_root(data_root: Path, document: dict) -> Path:
    batch = str(document.get("batch_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch) or batch in (".", ".."):
        raise EnvironmentError("unsafe environment batch identity")
    return data_root / "ptb" / KIND / batch


def worker_source_issues(repo_root: Path, ptb_root: Path, configuration: dict) -> list[str]:
    """This cluster cannot start jobs from the submit host's local temp trees."""
    if configuration.get("POST_TRAIN_BENCH_JOB_SCHEDULER") != "slurm":
        return []
    issues = []
    for label, path in (("top-level source/context records", repo_root), ("PTB WorkDir/entrypoint", ptb_root)):
        resolved = path.resolve()
        if any(resolved == local or resolved.is_relative_to(local) for local in (Path("/tmp"), Path("/var/tmp"))):
            issues.append(f"worker-local temporary {label}: {resolved}; deploy to the verified shared operator checkout")
    return issues


def report_directory(receipt: dict, job: dict) -> Path:
    if not is_operation(receipt) or receipt.get("kind") != KIND:
        raise EnvironmentError("receipt is not an environment operation")
    cell, job_id = str(job.get("cell_id", "")), str(job.get("job_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cell) or cell in (".", "..") or not job_id.isdigit():
        raise EnvironmentError("invalid environment job/cell identity")
    root = Path(str(receipt.get("environment_output_root", "")))
    if not root.is_absolute():
        raise EnvironmentError("environment receipt has no absolute output root")
    return root / cell / job_id


def _read_regular(root: Path, relative: str) -> bytes:
    # Reuse the retention layer's descriptor-relative, no-symlink reader.
    from awm.ptb_evidence_retention import _open_beneath
    with _open_beneath(root, Path(relative)) as stream:
        descriptor = stream.fileno()
        a = os.fstat(descriptor)
        raw = stream.read()
        b = os.fstat(descriptor)
    if not stat.S_ISREG(a.st_mode) or (a.st_ino, a.st_size, a.st_mtime_ns, a.st_ctime_ns) != (
        b.st_ino, b.st_size, b.st_mtime_ns, b.st_ctime_ns
    ) or len(raw) != a.st_size:
        raise EnvironmentError("environment evidence changed while reading")
    return raw


def _native_modules():
    """Load trusted checkout validators/program declarations; all imports are stdlib-only."""
    from awm import paths
    root = paths.REPO_ROOT / "third_party/PostTrainBench"
    modules = []
    for name, relative in (
        ("evidence", "src/eval/tasks/humaneval/official_evidence.py"),
        ("profile", "src/eval/tasks/humaneval/task.py"),
        ("probe", SOURCE_PATHS["probe"]),
    ):
        spec = importlib.util.spec_from_file_location("ptb_acceptance_" + name, root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def _validate_native_files(root, indexed, name, image, receipt):
    evidence, profile, probe = _native_modules()
    suffix = "home/task" if name == "opus_5.sif" else "result"
    prefix = f"official_eval/{name}/normal/{suffix}"

    def read(relative):
        if relative not in indexed:
            raise EnvironmentError("required native evidence is absent from raw index: " + relative)
        return indexed[relative]

    def document(relative):
        return evidence.strict_json(read(relative))

    observed = document(prefix + "/probe-result.json")
    if (observed.get("status") != "passed" or observed.get("real_model_called") is not False
            or observed.get("benchmark_programs_executed") is not False
            or any(observed.get(key) != image.get(key)
                   for key in ("cuda", "rows", "layout_observed", "native_revalidated"))
            or observed.get("dataset") != DATASET):
        raise EnvironmentError("native probe result differs from acceptance summary")
    request = document(prefix + "/request.json")
    runtime = document(prefix + "/python-runtime.json")
    metrics = document(prefix + "/metrics.json")
    programs = probe.programs()
    metadata = {"prompt": "def arrange(a,b):\n", "entry_point": "arrange",
                "test": "def check(fn):\n    assert fn(9,3)==[3,9]\n"}
    samples = [SimpleNamespace(id=case, input=case, target="synthetic target not executed",
                               metadata=metadata) for case in programs]
    contract = profile.selection_contract(samples)
    invocation = request.get("invocation", {})
    if (request.get("contract") != contract
            or invocation.get("attempt_id") != "environment-native-synthetic"
            or invocation.get("model") != "mockllm/model" or invocation.get("max_tokens") != 37
            or invocation.get("sandbox_helper_sha256") != receipt["operation"]["helper_sha256"]
            or runtime.get("helper_sha256") != invocation.get("sandbox_helper_sha256")
            or runtime.get("runtime_sha256") != invocation.get("sandbox_runtime_sha256")
            or not re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("runtime_sha256")))
            or runtime.get("limits") != profile.EXECUTION_LIMITS
            or invocation.get("sandbox_limits") != runtime.get("limits")
            or runtime.get("materialization", {}).get("source_image", {}).get("sha256") != image["sha256"]):
        raise EnvironmentError("native request/runtime differs from frozen synthetic operation")
    materialization = runtime.get("materialization", {})
    original_runtime = materialization.get("source_runtime")
    transported_runtime = runtime.get("runtime")
    if not isinstance(original_runtime, dict) or not isinstance(transported_runtime, dict):
        raise EnvironmentError("native runtime manifests are missing")
    original_sha = hashlib.sha256(json.dumps(original_runtime, sort_keys=True).encode()).hexdigest()
    transported_sha = hashlib.sha256(json.dumps(transported_runtime, sort_keys=True).encode()).hexdigest()
    materialization_sha = hashlib.sha256(
        (json.dumps(materialization, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    transport_binding = transported_runtime.get("materialization", {})
    if (original_sha != SOURCE_RUNTIME_SHA256[name]
            or runtime.get("source_runtime_sha256") != original_sha
            or materialization.get("source_runtime_sha256") != original_sha
            or transport_binding.get("source_runtime_sha256") != original_sha
            or transport_binding.get("sha256") != materialization_sha
            or transported_sha != runtime["runtime_sha256"]):
        raise EnvironmentError("native runtime manifest bytes do not match their frozen identities")
    publication = metrics.get("official_evidence", {})
    if (publication.get("kind") != "ptb-humaneval-inspect-v1"
            or publication.get("schema_version") != 1
            or publication.get("contract") != contract or publication.get("invocation") != invocation):
        raise EnvironmentError("native metrics publication identity mismatch")
    relative = publication.get("raw_log")
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts):
        raise EnvironmentError("unsafe native raw snapshot path")
    raw = read(prefix + "/" + relative)
    if (hashlib.sha256(raw).hexdigest() != publication.get("raw_sha256")
            or len(raw) != publication.get("raw_bytes")):
        raise EnvironmentError("native metrics raw snapshot binding differs")
    outer_prefix = f"official_eval/{name}/outer-timeout/{suffix}"
    admission = document(outer_prefix + "/outer-admitted.json")
    event = admission.get("event", {})
    outer_answer = probe.programs(outer_timeout=True)["invented-outer-timeout"]
    outer_code = metadata["prompt"] + outer_answer + "\n" + metadata["test"] + "\ncheck(arrange)"
    if (admission.get("observer") != "native-supervisor-admission"
            or admission.get("marker") != "PTB_OUTER_ADMITTED_68d273"
            or event != image["outer_timeout"].get("admission_event")
            or event.get("schema") != "ptb-python-admission-v1" or event.get("started") is not True
            or type(event.get("supervisor_pid")) is not int or event["supervisor_pid"] <= 0
            or event.get("attested_namespace_pid") != 1
            or event.get("code_sha256") != hashlib.sha256(outer_code.encode()).hexdigest()):
        raise EnvironmentError("outer native admission event is missing or unbound")
    log = evidence.strict_json(raw)
    checked = evidence.validate_log(log, contract, invocation)
    if (publication.get("validated") != checked or checked["accuracy"] != .4
            or checked["scored_samples"] != 5
            or any(metrics.get(key) != checked[key] for key in ("accuracy", "stderr"))):
        raise EnvironmentError("native metrics do not reconcile with five invented outcomes")
    rows = []
    for sample in log["samples"]:
        execution = sample["store"]["ptb_python_execution"][0]
        if sample["scores"]["verify"].get("answer") != programs[sample["id"]]:
            raise EnvironmentError("native answer differs from frozen invented program")
        rows.append({"id": sample["id"], "score": sample["scores"]["verify"]["value"],
                     "error": bool(sample.get("error")), "executions": 1,
                     "started": execution.get("started"), "outcome": execution.get("outcome"),
                     "error_category": execution.get("error_category"),
                     "cleanup_complete": execution.get("descendants_reaped") is True
                     and execution.get("monitor_reaped") is True and execution.get("cleanup_errors") == []})
    if sorted(rows, key=lambda row: row["id"]) != sorted(image["rows"], key=lambda row: row["id"]):
        raise EnvironmentError("native execution rows differ from acceptance summary")


def validate_acceptance(receipt: dict, job: dict, *, site_nodes: set[str], expected_uid: int,
                        directory: Path | None = None) -> dict:
    """Verify one receipt-backed synthetic acceptance; never a science verdict."""
    validate_operation(receipt)
    if receipt.get("kind") != KIND or job not in receipt.get("jobs", []):
        raise EnvironmentError("acceptance job is not in its operation receipt")
    if (type(expected_uid) is not int or expected_uid <= 0
            or any(not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source", {}).get(n, "")))
                   for n in ("top_commit", "ptb_commit"))):
        raise EnvironmentError("receipt lacks a frozen source or non-root identity")
    root = directory or report_directory(receipt, job)
    report = json.loads(_read_regular(root, "acceptance.json"))
    expected = {"schema_version": REPORT_SCHEMA, "job_id": str(job["job_id"]),
                "batch_id": receipt["batch_id"], "cell_id": job["cell_id"],
                "uid": expected_uid, "status": "passed", "errors": [], "scientific_result": False,
                "operation": KIND, "target": "humaneval"}
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in expected.items()):
        raise EnvironmentError("acceptance identity/status differs from receipt")
    if type(report["uid"]) is not int or expected_uid <= 0:
        raise EnvironmentError("acceptance must run as the frozen non-root identity")
    if report.get("node") not in effective_nodes(receipt, site_nodes):
        raise EnvironmentError("acceptance node differs from frozen placement")
    for name in ("top_commit", "ptb_commit"):
        if report.get("source", {}).get(name) != receipt.get("source", {}).get(name):
            raise EnvironmentError("acceptance source differs from frozen receipt")
    if report.get("probe_sources") != {
        name: receipt["operation"][f"{name}_sha256"] for name in SOURCE_PATHS
    } or report.get("dataset") != DATASET:
        raise EnvironmentError("acceptance probe/data binding differs")
    contract = receipt["contract"]
    images = {contract["container"]["name"] + ".sif": contract["container"]["sha256"],
              contract["evaluation_container"]["name"]: contract["evaluation_container"]["sha256"]}
    if not isinstance(report.get("images"), dict) or set(report["images"]) != set(images):
        raise EnvironmentError("acceptance requires both frozen images")
    expected_rows = {"invented-good": "C", "invented-wrong": "I", "invented-import": "I",
                     "invented-timeout": "I", "invented-private": "C"}
    for name, digest in images.items():
        image = report["images"][name]
        cuda = image.get("cuda", {})
        if (image.get("sha256") != digest or type(cuda.get("count")) is not int or cuda["count"] != 1
                or type(cuda.get("runtime_uid")) is not int or cuda["runtime_uid"] != expected_uid):
            raise EnvironmentError("acceptance image/GPU binding differs")
        if (not isinstance(cuda.get("names"), list) or len(cuda["names"]) != 1
                or not all(isinstance(n, str) and "H100" in n for n in cuda["names"])):
            raise EnvironmentError("acceptance requires the allocated H100")
        rows = image.get("rows")
        if (not isinstance(rows, list) or len(rows) != len(expected_rows)
                or any(not isinstance(r, dict) for r in rows)
                or {r.get("id"): r.get("score") for r in rows} != expected_rows
                or any(r.get("error") is not False or type(r.get("executions")) is not int
                       or r["executions"] != 1 or r.get("started") is not True
                       or r.get("cleanup_complete") is not True for r in rows)):
            raise EnvironmentError("native synthetic case evidence is incomplete")
        for row in rows:
            expected_outcome = ("timeout", "wall_timeout") if row["id"] == "invented-timeout" else (
                ("success", None) if row["score"] == "C" else ("program_failure", "program_exit"))
            if (row.get("outcome"), row.get("error_category")) != expected_outcome:
                raise EnvironmentError("native synthetic execution outcome differs")
        layout = image.get("layout", {})
        expected_role = "scientist" if name == "opus_5.sif" else "official-evaluator"
        if (layout.get("role") != expected_role
                or any(not isinstance(layout.get(k), str) or not Path(layout[k]).is_absolute()
                       for k in ("home", "cwd"))
                or image.get("layout_observed") != {k: layout[k] for k in ("home", "cwd")}):
            raise EnvironmentError("native production home/cwd layout is missing or differs")
        if expected_role == "scientist" and (layout["home"], layout["cwd"]) != ("/home/ben", "/home/ben/task"):
            raise EnvironmentError("scientist native layout differs")
        if expected_role == "official-evaluator" and not layout["cwd"].endswith("/src/eval/tasks/humaneval"):
            raise EnvironmentError("official evaluator native cwd differs")
        timeout = image.get("outer_timeout", {})
        elapsed = timeout.get("elapsed_since_admission")
        if (timeout.get("timed_out") is not True or timeout.get("cleanup_complete") is not True
                or timeout.get("passed") is not True or timeout.get("observed_native_sandbox") is not True
                or any(timeout.get(key) is not True for key in (
                    "admitted_before_timeout", "admitted_sandbox_live_at_timeout",
                    "admission_supervisor_live_at_timeout", "alarm_sent"))
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 < elapsed < 30
                or timeout.get("termination_grace_seconds") != 5
                or image.get("native_revalidated") is not True
                or any(image.get("normal_execution", {}).get(key) is not True
                       for key in ("passed", "cleanup_complete", "observed_native_sandbox"))):
            raise EnvironmentError("outer timeout/cleanup acceptance is missing")
        files = image.get("raw_files")
        if not isinstance(files, list) or not files:
            raise EnvironmentError("raw acceptance evidence is missing")
        names = set()
        indexed = {}
        for item in files:
            relative = item.get("path") if isinstance(item, dict) else None
            if not isinstance(relative, str) or relative in names:
                raise EnvironmentError("invalid or duplicate raw evidence path")
            if not relative.startswith(f"official_eval/{name}/"):
                raise EnvironmentError("raw evidence belongs to a different image")
            names.add(relative)
            raw = _read_regular(root, relative)
            if item.get("sha256") != hashlib.sha256(raw).hexdigest() or item.get("bytes") != len(raw):
                raise EnvironmentError("raw acceptance evidence changed")
            indexed[relative] = raw
        _validate_native_files(root, indexed, name, image, receipt)
    return report


def validate_readiness(manifest: dict, receipt: dict, *, site_nodes: set[str], ptb_commit: str,
                       completed_job_ids: set[str], directories: dict[str, Path] | None = None) -> list[dict]:
    """Check scoped admission against verified terminal probes, without opening an allowlist."""
    if manifest.get("contract", {}).get("task") != "humaneval" or is_operation(manifest):
        raise EnvironmentError("readiness applies only to a HumanEval scientific manifest")
    if receipt.get("source", {}).get("ptb_commit") != ptb_commit:
        raise EnvironmentError("acceptance PTB source differs from the proposed runtime")
    for key in ("container", "evaluation_container"):
        if manifest.get("contract", {}).get(key) != receipt.get("contract", {}).get(key):
            raise EnvironmentError("proposed image contract differs from acceptance")
    reports = []
    for job in receipt.get("jobs", []):
        if str(job["job_id"]) not in completed_job_ids:
            continue
        reports.append(validate_acceptance(receipt, job, site_nodes=site_nodes,
                                           expected_uid=receipt.get("environment_uid"),
                                           directory=(directories or {}).get(str(job["job_id"]))))
    admitted = {report["node"] for report in reports}
    if not effective_nodes(manifest, site_nodes) <= admitted:
        raise EnvironmentError("proposed HumanEval placement includes unaccepted nodes")
    return reports
