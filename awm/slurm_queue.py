"""Receipt-backed ownership registry and shared Slurm queue snapshots."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_QUEUE_ROOT = Path("/rmeng_data/robtang/slurm-queue")
DEFAULT_NODES = [f"slurm2-a3nodesetondem-{index}" for index in range(4)]
DEFAULT_SUBQUEUES = {
    "gangda_exp-protocol-evolve": {
        "branches": ["gangda_exp_protocol_evolve"],
        "gpu_limit": 16,
        "nodes": DEFAULT_NODES[:2],
    },
    "gangda_wma_evolve": {
        "branches": ["gangda_wma_evolve"],
        "gpu_limit": 16,
        "nodes": DEFAULT_NODES[2:],
    },
}
ACTIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
FAILURE_STATES = {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL"}


class QueueError(RuntimeError):
    pass


def queue_root() -> Path:
    return Path(os.environ.get("AWM_SLURM_QUEUE_ROOT", DEFAULT_QUEUE_ROOT)).resolve()


def default_registry_path() -> Path:
    return queue_root() / "registry.json"


def _default_subqueues() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "branches": list(config["branches"]),
            "gpu_limit": config["gpu_limit"],
            "nodes": list(config["nodes"]),
        }
        for name, config in DEFAULT_SUBQUEUES.items()
    }


def _default_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "queue_name": "gangda",
        "owner": "robtang_google_com",
        "scope": {
            "partition": "ptb-a3",
            "reservation": "robtang-ptb-a3",
            "nodes": DEFAULT_NODES,
            "gpus_per_node": 8,
        },
        "subqueues": _default_subqueues(),
        "sources": [],
    }


def _validate_subqueues(data: dict[str, Any], path: Path | None = None) -> None:
    where = f": {path}" if path else ""
    subqueues = data.get("subqueues")
    if not isinstance(subqueues, dict) or not subqueues:
        raise QueueError(f"ownership registry has no subqueue map{where}")
    scope = data.get("scope") or {}
    scope_nodes = set(scope.get("nodes") or DEFAULT_NODES)
    gpus_per_node = int(scope.get("gpus_per_node", 8))
    assigned_nodes: set[str] = set()
    assigned_gpus = 0
    for name, config in subqueues.items():
        if not isinstance(name, str) or not name or not isinstance(config, dict):
            raise QueueError(f"invalid subqueue entry {name!r}{where}")
        nodes = config.get("nodes")
        branches = config.get("branches")
        gpu_limit = config.get("gpu_limit")
        if (
            not isinstance(nodes, list)
            or not nodes
            or not all(isinstance(node, str) and node for node in nodes)
        ):
            raise QueueError(f"subqueue {name} has invalid nodes{where}")
        if (
            not isinstance(branches, list)
            or not branches
            or not all(isinstance(branch, str) and branch for branch in branches)
        ):
            raise QueueError(f"subqueue {name} has invalid branches{where}")
        if not isinstance(gpu_limit, int) or gpu_limit <= 0:
            raise QueueError(f"subqueue {name} has invalid gpu_limit{where}")
        if set(nodes) - scope_nodes:
            raise QueueError(f"subqueue {name} names nodes outside the gangda scope{where}")
        if assigned_nodes & set(nodes):
            raise QueueError(f"subqueue {name} overlaps another subqueue's nodes{where}")
        expected = len(nodes) * gpus_per_node
        if gpu_limit != expected:
            raise QueueError(
                f"subqueue {name} gpu_limit={gpu_limit} does not match "
                f"{len(nodes)} node(s) x {gpus_per_node} GPU(s){where}"
            )
        assigned_nodes.update(nodes)
        assigned_gpus += gpu_limit
    total_gpus = len(scope_nodes) * gpus_per_node
    if assigned_nodes != scope_nodes or assigned_gpus != total_gpus:
        raise QueueError(
            f"subqueues cover {assigned_gpus}/{total_gpus} GPUs and "
            f"{len(assigned_nodes)}/{len(scope_nodes)} nodes{where}"
        )


def load_registry(path: Path | None = None) -> dict[str, Any]:
    path = (path or default_registry_path()).resolve()
    if not path.exists():
        return _default_registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QueueError(f"cannot read ownership registry {path}: {exc}") from exc
    if data.get("schema_version") != 1 or not isinstance(data.get("sources"), list):
        raise QueueError(f"invalid ownership registry: {path}")
    data.setdefault("subqueues", _default_subqueues())
    _validate_subqueues(data, path)
    return data


def subqueue_config(data: dict[str, Any], name: str) -> dict[str, Any]:
    _validate_subqueues(data)
    config = data["subqueues"].get(name)
    if not isinstance(config, dict):
        raise QueueError(f"unknown gangda subqueue: {name}")
    return config


def subqueue_for_branch(data: dict[str, Any], branch: str) -> str | None:
    matches = [
        name
        for name, config in data["subqueues"].items()
        if branch in config.get("branches", [])
    ]
    if len(matches) > 1:
        raise QueueError(f"branch {branch} belongs to multiple gangda subqueues")
    return matches[0] if matches else None


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _update_registry(source: dict[str, Any], path: Path | None = None) -> Path:
    path = (path or default_registry_path()).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_registry(path)
        data.setdefault("queue_name", "gangda")
        sources = [item for item in data["sources"] if item.get("id") != source["id"]]
        sources.append(source)
        data["sources"] = sorted(sources, key=lambda item: str(item.get("id")))
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def register_receipt(
    receipt_path: Path, *, label: str | None = None, registry_path: Path | None = None
) -> Path:
    receipt_path = receipt_path.resolve()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QueueError(f"cannot read receipt {receipt_path}: {exc}") from exc
    jobs = receipt.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise QueueError(f"receipt has no jobs: {receipt_path}")
    normalized_jobs = []
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        job_name = str(job.get("job_name", ""))
        if not job_id.isdigit() or not job_name:
            raise QueueError(f"receipt contains an invalid job: {job!r}")
        normalized_jobs.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "cell_id": str(job.get("cell_id", "")),
            }
        )
    batch_id = str(receipt.get("batch_id", receipt_path.parent.name))
    receipt_kind = str(receipt.get("kind", ""))
    default_label = f"{batch_id} [{receipt_kind}]" if receipt_kind else batch_id
    registry = load_registry(registry_path)
    declared_subqueue = str(
        receipt.get("subqueue") or (receipt.get("site") or {}).get("subqueue") or ""
    )
    branch = str((receipt.get("ownership") or {}).get("branch", ""))
    inferred_subqueue = subqueue_for_branch(registry, branch) if branch else None
    if declared_subqueue:
        subqueue_config(registry, declared_subqueue)
    if declared_subqueue and inferred_subqueue and declared_subqueue != inferred_subqueue:
        raise QueueError(
            f"receipt declares subqueue {declared_subqueue} but branch {branch} belongs to "
            f"{inferred_subqueue}"
        )
    subqueue = declared_subqueue or inferred_subqueue
    source = {
        "id": f"receipt:{receipt_path}",
        "kind": "receipt",
        "label": label or default_label,
        "path": str(receipt_path),
        "batch_id": batch_id,
        "receipt_kind": receipt_kind,
        "manifest": str(receipt.get("manifest", "")),
        "spec": str((receipt.get("ownership") or {}).get("spec", "")),
        "jobs": normalized_jobs,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    if subqueue:
        source["subqueue"] = subqueue
    return _update_registry(source, registry_path)


def unregister_receipt(receipt_path: Path, *, registry_path: Path | None = None) -> Path:
    receipt_path = receipt_path.resolve()
    registry_path = (registry_path or default_registry_path()).resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    source_id = f"receipt:{receipt_path}"
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_registry(registry_path)
        sources = [source for source in data["sources"] if source.get("id") != source_id]
        if len(sources) == len(data["sources"]):
            raise QueueError(f"receipt is not registered: {receipt_path}")
        data["sources"] = sources
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(registry_path, json.dumps(data, indent=2, sort_keys=True) + "\n")
    return registry_path


def _job_record(job_id: str) -> dict[str, str]:
    if not job_id.isdigit():
        raise QueueError(f"invalid Slurm job id: {job_id}")
    result = subprocess.run(
        ["scontrol", "show", "job", job_id, "-o"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise QueueError(result.stderr.strip() or f"cannot inspect Slurm job {job_id}")
    fields = {}
    for field in result.stdout.split():
        if "=" in field:
            key, value = field.split("=", 1)
            fields[key] = value
    if not fields.get("JobName"):
        raise QueueError(f"Slurm job {job_id} has no name")
    return {
        "job_id": job_id,
        "job_name": fields["JobName"],
        "work_dir": fields.get("WorkDir", ""),
        "stdout": fields.get("StdOut", ""),
    }


def register_job(
    job_id: str,
    *,
    label: str,
    source_id: str | None = None,
    subqueue: str | None = None,
    registry_path: Path | None = None,
) -> Path:
    if subqueue:
        subqueue_config(load_registry(registry_path), subqueue)
    job = _job_record(job_id)
    source = {
        "id": source_id or f"job:{job_id}",
        "kind": "explicit_jobs",
        "label": label,
        "jobs": [job],
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    if subqueue:
        source["subqueue"] = subqueue
    return _update_registry(source, registry_path)


def _command(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise QueueError(result.stderr.strip() or f"command failed: {' '.join(command)}")
    return result.stdout


def _live_jobs() -> dict[str, dict[str, str]]:
    output = _command(["squeue", "-h", "-o", "%i|%T|%R|%N|%j|%M|%u|%P"])
    rows = {}
    for line in output.splitlines():
        parts = line.split("|", 7)
        if len(parts) != 8:
            continue
        job_id, state, reason, nodes, name, elapsed, user, partition = parts
        rows[job_id] = {
            "job_id": job_id,
            "state": state,
            "reason": reason,
            "nodes": nodes,
            "job_name": name,
            "elapsed": elapsed,
            "user": user,
            "partition": partition,
        }
    return rows


def _terminal_jobs(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    output = _command(
        [
            "sacct",
            "-nX",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State,Elapsed,NodeList",
            "--parsable2",
        ]
    )
    rows = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 4 or parts[0] not in job_ids:
            continue
        rows[parts[0]] = {
            "job_id": parts[0],
            "state": parts[1],
            "elapsed": parts[2],
            "nodes": parts[3],
            "reason": "",
        }
    return rows


def _node_status(nodes: list[str], gpus_per_node: int) -> list[dict[str, Any]]:
    statuses = []
    for node in nodes:
        output = _command(["scontrol", "show", "node", node, "-o"])
        fields = {}
        for field in output.split():
            if "=" in field:
                key, value = field.split("=", 1)
                fields[key] = value
        allocated = 0
        for item in fields.get("AllocTRES", "").split(","):
            if item.startswith("gres/gpu="):
                allocated = int(item.split("=", 1)[1])
        statuses.append(
            {
                "node": node,
                "state": fields.get("State", "UNKNOWN"),
                "gpus_allocated": allocated,
                "gpus_total": gpus_per_node,
            }
        )
    return statuses


def collect_snapshot(registry_path: Path | None = None) -> dict[str, Any]:
    registry_path = (registry_path or default_registry_path()).resolve()
    registry = load_registry(registry_path)
    expected: dict[str, dict[str, str]] = {}
    source_for_job: dict[str, str] = {}
    for source in registry["sources"]:
        for job in source.get("jobs", []):
            job_id = str(job["job_id"])
            if job_id in expected and source_for_job[job_id] != source["id"]:
                raise QueueError(f"job {job_id} is registered by multiple sources")
            expected[job_id] = job
            source_for_job[job_id] = source["id"]

    live = _live_jobs()
    terminal = _terminal_jobs([job_id for job_id in expected if job_id not in live])
    scope = registry.get("scope") or {}
    nodes = list(scope.get("nodes") or DEFAULT_NODES)
    node_set = set(nodes)
    gpus_per_node = int(scope.get("gpus_per_node", 8))

    unknown = []
    for job_id, job in live.items():
        assigned = {part for part in job["nodes"].split(",") if part}
        if assigned & node_set and job_id not in expected:
            unknown.append(job)

    sources = []
    name_mismatches = []
    for source in registry["sources"]:
        jobs = []
        for expected_job in source.get("jobs", []):
            job_id = str(expected_job["job_id"])
            current = dict(live.get(job_id) or terminal.get(job_id) or {"state": "UNKNOWN"})
            current.update(
                {
                    "job_id": job_id,
                    "cell_id": expected_job.get("cell_id", ""),
                    "expected_name": expected_job.get("job_name", ""),
                    "work_dir": expected_job.get("work_dir", ""),
                    "stdout": expected_job.get("stdout", ""),
                }
            )
            actual_name = current.get("job_name", "")
            if actual_name and actual_name != expected_job.get("job_name"):
                name_mismatches.append(
                    {
                        "job_id": job_id,
                        "expected": expected_job.get("job_name", ""),
                        "actual": actual_name,
                    }
                )
            jobs.append(current)
        counts = Counter(str(job.get("state", "UNKNOWN")) for job in jobs)
        sources.append(
            {
                "id": source["id"],
                "label": source.get("label", source["id"]),
                "kind": source.get("kind", "unknown"),
                "path": source.get("path", ""),
                "batch_id": source.get("batch_id", ""),
                "receipt_kind": source.get("receipt_kind", ""),
                "registered_at": source.get("registered_at", ""),
                "manifest": source.get("manifest", ""),
                "spec": source.get("spec", ""),
                "subqueue": source.get("subqueue", ""),
                "counts": dict(sorted(counts.items())),
                "active": sum(count for state, count in counts.items() if state in ACTIVE_STATES),
                "jobs": jobs,
            }
        )

    node_status = _node_status(nodes, gpus_per_node)
    node_by_name = {node["node"]: node for node in node_status}
    subqueues = []
    for name, config in registry["subqueues"].items():
        subqueue_nodes = [node_by_name[node] for node in config["nodes"]]
        subqueues.append(
            {
                "name": name,
                "branches": list(config["branches"]),
                "gpu_limit": config["gpu_limit"],
                "nodes": subqueue_nodes,
                "gpus_allocated": sum(node["gpus_allocated"] for node in subqueue_nodes),
                "gpus_total": config["gpu_limit"],
                "active_registered": sum(
                    source["active"] for source in sources if source["subqueue"] == name
                ),
            }
        )
    snapshot = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(registry_path),
        "queue_name": registry.get("queue_name", "gangda"),
        "owner": registry.get("owner", ""),
        "scope": scope,
        "ownership_ok": not unknown and not name_mismatches,
        "unknown_jobs": sorted(unknown, key=lambda item: item["job_id"]),
        "name_mismatches": name_mismatches,
        "nodes": node_status,
        "gpus_allocated": sum(node["gpus_allocated"] for node in node_status),
        "gpus_total": sum(node["gpus_total"] for node in node_status),
        "subqueues": subqueues,
        "sources": sources,
    }
    return snapshot


def _state_key(state: object) -> str:
    return str(state or "UNKNOWN").split()[0].rstrip("+")


def _job_line(job: dict[str, Any]) -> str:
    cell = f" cell={job['cell_id']}" if job.get("cell_id") else ""
    node = job.get("nodes") or "-"
    reason = (
        f" reason={job['reason']}"
        if _state_key(job.get("state")) == "PENDING" and job.get("reason")
        else ""
    )
    return (
        f"    {job['job_id']} {job.get('state', 'UNKNOWN')} node={node}"
        f" elapsed={job.get('elapsed', '-')}{cell}{reason}"
    )


def select_subqueue(snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    selected = next((item for item in snapshot.get("subqueues", []) if item["name"] == name), None)
    if selected is None:
        raise QueueError(f"unknown gangda subqueue: {name}")
    node_names = {node["node"] for node in selected["nodes"]}
    view = dict(snapshot)
    view["selected_subqueue"] = name
    view["nodes"] = selected["nodes"]
    view["gpus_allocated"] = selected["gpus_allocated"]
    view["gpus_total"] = selected["gpus_total"]
    view["sources"] = [
        source for source in snapshot["sources"] if source.get("subqueue") == name
    ]
    view["unknown_jobs"] = [
        job
        for job in snapshot["unknown_jobs"]
        if {part for part in job.get("nodes", "").split(",") if part} & node_names
    ]
    return view


def render_snapshot(
    snapshot: dict[str, Any], *, include_jobs: bool = True, subqueue: str | None = None
) -> str:
    """Render the operational view: only jobs that are active now."""
    if subqueue:
        snapshot = select_subqueue(snapshot, subqueue)
    verdict = "OK" if snapshot["ownership_ok"] else "FAIL"
    lines = [
        f"updated={snapshot['updated_at']}",
        f"QUEUE {snapshot.get('queue_name', 'gangda')}",
    ]
    if snapshot.get("selected_subqueue"):
        lines.append(f"SUBQUEUE {snapshot['selected_subqueue']}")
    lines.extend(
        [
            f"OWNERSHIP {verdict}  owner={snapshot['owner']}",
            f"GPUS {snapshot['gpus_allocated']}/{snapshot['gpus_total']} allocated",
        ]
    )
    guard = snapshot.get("guard") or {}
    if guard.get("enabled"):
        lines.insert(2, f"GUARD ON  unknown_grace={guard['grace_seconds']}s")
    if not snapshot.get("selected_subqueue") and snapshot.get("subqueues"):
        lines.append("SUBQUEUES")
        for item in snapshot["subqueues"]:
            nodes = ",".join(node["node"] for node in item["nodes"])
            lines.append(
                f"  {item['name']}: GPUS {item['gpus_allocated']}/{item['gpus_total']} "
                f"allocated registered_active={item['active_registered']} nodes={nodes}"
            )
    for node in snapshot["nodes"]:
        lines.append(
            f"  {node['node']}: {node['gpus_allocated']}/{node['gpus_total']} state={node['state']}"
        )
    lines.append("SOURCES")
    active_sources = [source for source in snapshot["sources"] if source["active"]]
    for source in active_sources:
        counts = " ".join(
            f"{state}={count}"
            for state, count in source["counts"].items()
            if state in ACTIVE_STATES
        )
        queue_tag = source.get("subqueue") or "unassigned"
        lines.append(f"  [{queue_tag}] {source['label']}: {counts}")
        if include_jobs:
            for job in source["jobs"]:
                if _state_key(job.get("state")) in ACTIVE_STATES:
                    lines.append(_job_line(job))
    if not active_sources:
        lines.append("  no active registered jobs")
    if snapshot["unknown_jobs"]:
        lines.append("UNKNOWN JOBS ON OWNED NODES")
        for job in snapshot["unknown_jobs"]:
            lines.append(
                f"  {job['job_id']} {job['state']} node={job['nodes']} name={job['job_name']}"
            )
    if snapshot["name_mismatches"]:
        lines.append("NAME MISMATCHES")
        for mismatch in snapshot["name_mismatches"]:
            lines.append(
                f"  {mismatch['job_id']} expected={mismatch['expected']} "
                f"actual={mismatch['actual']}"
            )
    return "\n".join(lines) + "\n"


def _has_validated_ptb_result(job_id: str) -> bool:
    from awm import ptb_experiments

    result_dir = ptb_experiments.result_for_job(job_id)
    return result_dir is not None and not ptb_experiments.audit_result(result_dir)


def failure_records(snapshot: dict[str, Any], *, include_resolved: bool = False) -> list[dict]:
    """Return terminal failures, suppressing ones replaced by a later healthy retry."""
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}
    for source in snapshot["sources"]:
        batch_id = str(source.get("batch_id", ""))
        registered_at = str(source.get("registered_at", ""))
        for job in source["jobs"]:
            cell_id = str(job.get("cell_id", ""))
            if batch_id and cell_id:
                candidates.setdefault((batch_id, cell_id), []).append((registered_at, source, job))

    failures = []
    for source in snapshot["sources"]:
        source_time = str(source.get("registered_at", ""))
        batch_id = str(source.get("batch_id", ""))
        for job in source["jobs"]:
            state = _state_key(job.get("state"))
            if state not in FAILURE_STATES:
                continue
            cell_id = str(job.get("cell_id", ""))
            replacement = (
                {"source": "validated PTB result", "job_id": job["job_id"], "state": "COMPLETE"}
                if _has_validated_ptb_result(str(job["job_id"]))
                else None
            )
            if replacement is None:
                for candidate_time, candidate_source, candidate_job in candidates.get(
                    (batch_id, cell_id), []
                ):
                    candidate_state = _state_key(candidate_job.get("state"))
                    if candidate_time > source_time and candidate_state in (
                        ACTIVE_STATES | {"COMPLETED"}
                    ):
                        replacement = {
                            "source": candidate_source["label"],
                            "job_id": candidate_job["job_id"],
                            "state": candidate_job.get("state", "UNKNOWN"),
                        }
                        break
            record = {
                "source": source["label"],
                "source_path": source.get("path", ""),
                "job": job,
                "resolved": replacement is not None,
                "replacement": replacement,
            }
            if include_resolved or not record["resolved"]:
                failures.append(record)
    return failures


def render_failures(snapshot: dict[str, Any], *, include_resolved: bool = False) -> str:
    failures = failure_records(snapshot, include_resolved=include_resolved)
    lines = [
        f"updated={snapshot['updated_at']}",
        f"QUEUE {snapshot.get('queue_name', 'gangda')} FAILURES",
    ]
    if snapshot["unknown_jobs"] or snapshot["name_mismatches"]:
        lines.append("OWNERSHIP FAIL")
    if not failures:
        lines.append("NO UNRESOLVED FAILURES")
        return "\n".join(lines) + "\n"
    for record in failures:
        job = record["job"]
        suffix = ""
        if record["resolved"]:
            replacement = record["replacement"]
            suffix = (
                f" resolved_by={replacement['job_id']}:{replacement['state']}"
                f" source={replacement['source']}"
            )
        lines.append(
            f"  {job['job_id']} {job.get('state', 'UNKNOWN')} source={record['source']}"
            f" cell={job.get('cell_id') or '-'} elapsed={job.get('elapsed', '-')}{suffix}"
        )
    return "\n".join(lines) + "\n"


def render_history(snapshot: dict[str, Any], *, include_jobs: bool = True) -> str:
    lines = [
        f"updated={snapshot['updated_at']}",
        f"QUEUE {snapshot.get('queue_name', 'gangda')} HISTORY",
    ]
    historical_sources = []
    for source in snapshot["sources"]:
        terminal_jobs = [
            job for job in source["jobs"] if _state_key(job.get("state")) not in ACTIVE_STATES
        ]
        if terminal_jobs:
            historical_sources.append((source, terminal_jobs))
    if not historical_sources:
        lines.append("NO TERMINAL HISTORY")
        return "\n".join(lines) + "\n"
    for source, jobs in historical_sources:
        counts = Counter(str(job.get("state", "UNKNOWN")) for job in jobs)
        summary = " ".join(f"{state}={count}" for state, count in sorted(counts.items()))
        lines.append(f"  {source['label']}: {summary}")
        if source.get("path"):
            lines.append(f"    receipt={source['path']}")
        if source.get("manifest"):
            lines.append(f"    manifest={source['manifest']}")
        if source.get("spec"):
            lines.append(f"    spec={source['spec']}")
        if include_jobs:
            lines.extend(_job_line(job) for job in jobs)
    return "\n".join(lines) + "\n"


def explain_job(snapshot: dict[str, Any], job_id: str) -> dict[str, Any]:
    for source in snapshot["sources"]:
        for job in source["jobs"]:
            if str(job["job_id"]) != str(job_id):
                continue
            explanation: dict[str, Any] = {
                "queue": snapshot.get("queue_name", "gangda"),
                "source": {
                    key: source.get(key, "")
                    for key in ("label", "kind", "path", "batch_id", "manifest", "spec")
                },
                "job": job,
                "cell": {},
                "frozen_source": {},
                "result": {},
            }
            receipt_path = Path(str(source.get("path", "")))
            if receipt_path.is_file():
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise QueueError(f"cannot read receipt {receipt_path}: {exc}") from exc
                explanation["frozen_source"] = receipt.get("source") or {}
                cell_id = str(job.get("cell_id", ""))
                explanation["cell"] = next(
                    (
                        cell
                        for cell in receipt.get("cells", [])
                        if str(cell.get("id", "")) == cell_id
                    ),
                    {},
                )
            from awm import ptb_experiments, ptb_results

            result_dir = ptb_experiments.result_for_job(str(job_id))
            if result_dir:
                metrics = ptb_results._read_json(result_dir / "metrics.json")
                issues = ptb_experiments.audit_result(result_dir)
                explanation["result"] = {
                    "path": str(result_dir),
                    "complete": not issues,
                    "issues": issues,
                    "accuracy": metrics.get("accuracy"),
                    "stderr": metrics.get("stderr"),
                    "judge_flags": ptb_results.judge_flags(result_dir),
                }
            return explanation
    raise QueueError(f"job is not registered in gangda: {job_id}")


def render_job_explanation(explanation: dict[str, Any]) -> str:
    job = explanation["job"]
    source = explanation["source"]
    cell = explanation["cell"]
    frozen = explanation["frozen_source"]
    result = explanation["result"]
    lines = [
        f"JOB {job['job_id']}",
        (
            f"  state={job.get('state', 'UNKNOWN')} node={job.get('nodes') or '-'} "
            f"elapsed={job.get('elapsed', '-')}"
        ),
        f"  source={source.get('label') or '-'}",
        f"  cell={job.get('cell_id') or '-'}",
    ]
    for key in (
        "task",
        "base_model",
        "agent",
        "agent_model",
        "effort",
        "context_tokens",
        "replicate",
    ):
        if key in cell:
            lines.append(f"  {key}={cell[key]}")
    lines.append(f"  slurm_name={job.get('expected_name') or job.get('job_name') or '-'}")
    for key in ("path", "manifest", "spec"):
        if source.get(key):
            lines.append(f"  {key}={source[key]}")
    for key in ("top_commit", "ptb_commit"):
        if frozen.get(key):
            lines.append(f"  {key}={frozen[key]}")
    if result:
        lines.append(f"  result={result['path']}")
        lines.append(f"  result_complete={str(result['complete']).lower()}")
        if result.get("accuracy") is not None:
            lines.append(f"  accuracy={result['accuracy']}")
        lines.append(f"  judge_flags={','.join(result['judge_flags']) or 'clean'}")
        if result.get("issues"):
            lines.append(f"  result_issues={'; '.join(result['issues'])}")
    if job.get("work_dir"):
        lines.append(f"  work_dir={job['work_dir']}")
    if job.get("stdout"):
        lines.append(f"  stdout={job['stdout']}")
    return "\n".join(lines) + "\n"


def write_snapshot(snapshot: dict[str, Any], root: Path | None = None) -> None:
    root = (root or queue_root()).resolve()
    _atomic_write(root / "current.json", json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    _atomic_write(root / "current.txt", render_snapshot(snapshot))


def _enforcement_due(
    snapshot: dict[str, Any], seen: dict[str, float], *, now: float, grace: int
) -> tuple[dict[str, float], list[str]]:
    unknown_ids = {str(job["job_id"]) for job in snapshot["unknown_jobs"]}
    unknown_ids.update(str(item["job_id"]) for item in snapshot["name_mismatches"])
    current_seen = {job_id: seen.get(job_id, now) for job_id in unknown_ids}
    due = sorted(job_id for job_id, first_seen in current_seen.items() if now - first_seen >= grace)
    return current_seen, due


def _enforce_unknown(snapshot: dict[str, Any], root: Path, grace: int) -> list[str]:
    seen_path = root / "unknown-seen.json"
    try:
        seen = json.loads(seen_path.read_text(encoding="utf-8")) if seen_path.is_file() else {}
    except (OSError, ValueError):
        seen = {}
    if not isinstance(seen, dict):
        seen = {}
    now = time.time()
    current_seen, due = _enforcement_due(snapshot, seen, now=now, grace=grace)
    _atomic_write(seen_path, json.dumps(current_seen, indent=2, sort_keys=True) + "\n")
    if not due:
        return []
    result = subprocess.run(
        ["sudo", "-n", "scancel", *due],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise QueueError(
            result.stderr.strip() or f"ownership guard could not cancel: {', '.join(due)}"
        )
    with (root / "enforcement.log").open("a", encoding="utf-8") as stream:
        stream.write(
            f"{datetime.now(timezone.utc).isoformat()} cancelled unknown jobs: {','.join(due)}\n"
        )
    return due


def monitor_loop(
    interval: int,
    registry_path: Path | None = None,
    enforce_unknown_after: int | None = None,
) -> None:
    root = queue_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "monitor.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QueueError("queue monitor is already running") from exc
        _atomic_write(root / "monitor.pid", f"{os.getpid()}\n")
        while True:
            try:
                snapshot = collect_snapshot(registry_path)
                snapshot["guard"] = {
                    "enabled": enforce_unknown_after is not None,
                    "grace_seconds": enforce_unknown_after,
                }
                write_snapshot(snapshot, root)
                if enforce_unknown_after is not None:
                    _enforce_unknown(snapshot, root, enforce_unknown_after)
                _atomic_write(root / "monitor.error", "")
            except (OSError, QueueError, KeyError, TypeError, ValueError) as exc:
                # Keep the shared monitor alive across transient Slurm and filesystem errors.
                _atomic_write(
                    root / "monitor.error",
                    f"{datetime.now(timezone.utc).isoformat()} {type(exc).__name__}: {exc}\n",
                )
            time.sleep(interval)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def start_monitor(
    interval: int,
    registry_path: Path | None = None,
    enforce_unknown_after: int | None = None,
) -> int:
    if interval < 1:
        raise QueueError("monitor interval must be positive")
    if enforce_unknown_after is not None and enforce_unknown_after < 0:
        raise QueueError("ownership enforcement grace must not be negative")
    root = queue_root()
    root.mkdir(parents=True, exist_ok=True)
    pid_path = root / "monitor.pid"
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if _pid_is_alive(pid):
            return pid
    command = [sys.executable, "-m", "awm.slurm_queue", "monitor-loop", "--interval", str(interval)]
    if registry_path:
        command.extend(["--registry", str(registry_path.resolve())])
    if enforce_unknown_after is not None:
        command.extend(["--enforce-unknown-after", str(enforce_unknown_after)])
    with (root / "monitor.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    for _ in range(50):
        if pid_path.is_file() and pid_path.read_text(encoding="utf-8").strip() == str(process.pid):
            return process.pid
        if process.poll() is not None:
            raise QueueError(f"queue monitor exited with code {process.returncode}")
        time.sleep(0.1)
    raise QueueError("queue monitor did not create its pid file")


def monitor_status() -> tuple[bool, int | None]:
    pid_path = queue_root() / "monitor.pid"
    if not pid_path.is_file():
        return False, None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return False, None
    return _pid_is_alive(pid), pid


def module_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    monitor = sub.add_parser("monitor-loop")
    monitor.add_argument("--interval", type=int, default=15)
    monitor.add_argument("--registry", type=Path)
    monitor.add_argument("--enforce-unknown-after", type=int)
    args = parser.parse_args(argv)
    if args.command == "monitor-loop":
        monitor_loop(args.interval, args.registry, args.enforce_unknown_after)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(module_main())
