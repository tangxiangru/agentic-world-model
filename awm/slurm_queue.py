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


class QueueError(RuntimeError):
    pass


def queue_root() -> Path:
    return Path(os.environ.get("AWM_SLURM_QUEUE_ROOT", DEFAULT_QUEUE_ROOT)).resolve()


def default_registry_path() -> Path:
    return queue_root() / "registry.json"


def _default_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "owner": "robtang_google_com",
        "scope": {
            "partition": "ptb-a3",
            "reservation": "robtang-ptb-a3",
            "nodes": DEFAULT_NODES,
            "gpus_per_node": 8,
        },
        "sources": [],
    }


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
    return data


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
    source = {
        "id": f"receipt:{receipt_path}",
        "kind": "receipt",
        "label": label or batch_id,
        "path": str(receipt_path),
        "batch_id": batch_id,
        "jobs": normalized_jobs,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    return _update_registry(source, registry_path)


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
    registry_path: Path | None = None,
) -> Path:
    job = _job_record(job_id)
    source = {
        "id": source_id or f"job:{job_id}",
        "kind": "explicit_jobs",
        "label": label,
        "jobs": [job],
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
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
    active_states = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
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
                "counts": dict(sorted(counts.items())),
                "active": sum(count for state, count in counts.items() if state in active_states),
                "jobs": jobs,
            }
        )

    node_status = _node_status(nodes, gpus_per_node)
    snapshot = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(registry_path),
        "owner": registry.get("owner", ""),
        "scope": scope,
        "ownership_ok": not unknown and not name_mismatches,
        "unknown_jobs": sorted(unknown, key=lambda item: item["job_id"]),
        "name_mismatches": name_mismatches,
        "nodes": node_status,
        "gpus_allocated": sum(node["gpus_allocated"] for node in node_status),
        "gpus_total": sum(node["gpus_total"] for node in node_status),
        "sources": sources,
    }
    return snapshot


def render_snapshot(snapshot: dict[str, Any], *, include_jobs: bool = True) -> str:
    verdict = "OK" if snapshot["ownership_ok"] else "FAIL"
    lines = [
        f"updated={snapshot['updated_at']}",
        f"OWNERSHIP {verdict}  owner={snapshot['owner']}",
        f"GPUS {snapshot['gpus_allocated']}/{snapshot['gpus_total']} allocated",
    ]
    guard = snapshot.get("guard") or {}
    if guard.get("enabled"):
        lines.insert(2, f"GUARD ON  unknown_grace={guard['grace_seconds']}s")
    for node in snapshot["nodes"]:
        lines.append(
            f"  {node['node']}: {node['gpus_allocated']}/{node['gpus_total']} state={node['state']}"
        )
    lines.append("SOURCES")
    for source in snapshot["sources"]:
        counts = " ".join(f"{state}={count}" for state, count in source["counts"].items())
        lines.append(f"  {source['label']}: {counts or 'no jobs'}")
        if include_jobs:
            for job in source["jobs"]:
                cell = f" cell={job['cell_id']}" if job.get("cell_id") else ""
                node = job.get("nodes") or "-"
                reason = (
                    f" reason={job['reason']}"
                    if job.get("state") == "PENDING" and job.get("reason")
                    else ""
                )
                lines.append(
                    f"    {job['job_id']} {job.get('state', 'UNKNOWN')} node={node}"
                    f" elapsed={job.get('elapsed', '-')}{cell}{reason}"
                )
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
