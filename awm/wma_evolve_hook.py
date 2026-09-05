"""Event-driven monitor for the online WMA evolution loop.

The daemon polls slowly, freezes validator-clean PTB completion windows, and
launches a read-only Claude Code analysis.  Queue mutation remains an operator
action; this module never submits or cancels Slurm jobs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_REPO = Path("/home/robtang_google_com/gangda_workspace/agentic-world-model")
DEFAULT_REGISTRY = Path("/rmeng_data/robtang/slurm-queue/registry.json")
DEFAULT_CURRENT = Path("/rmeng_data/robtang/slurm-queue/current.json")
DEFAULT_STATE_DIR = Path("/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve")
DEFAULT_SUBQUEUE = "gangda_wma_evolve"
DEFAULT_BRANCH = "gangda_wma_evolve"
DEFAULT_PREFIX = "wma-"
EXPECTED_NODES = "slurm2-a3nodesetondem-[2-3]"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@dataclass(frozen=True)
class ManifestRecord:
    batch_id: str
    manifest: Path
    receipt: Path
    registered_at: str


def discover_manifests(
    registry_path: Path,
    *,
    subqueue: str = DEFAULT_SUBQUEUE,
    prefix: str = DEFAULT_PREFIX,
) -> list[ManifestRecord]:
    """Resolve the newest receipt for every in-scope batch from the registry."""
    registry = _load_json(registry_path)
    newest: dict[str, ManifestRecord] = {}
    for source in registry.get("sources", []):
        if source.get("kind") != "receipt":
            continue
        label = str(source.get("label", ""))
        if not label.startswith(prefix):
            continue
        source_id = str(source.get("id", ""))
        if not source_id.startswith("receipt:"):
            continue
        receipt_path = Path(source_id.removeprefix("receipt:"))
        try:
            receipt = _load_json(receipt_path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if receipt.get("subqueue") != subqueue:
            continue
        batch_id = str(receipt.get("batch_id", ""))
        manifest = Path(str(receipt.get("manifest", "")))
        if not batch_id.startswith(prefix) or not manifest.is_file():
            continue
        stamp = str(source.get("registered_at") or receipt.get("submitted_at") or "")
        record = ManifestRecord(batch_id, manifest, receipt_path, stamp)
        previous = newest.get(batch_id)
        if previous is None or record.registered_at > previous.registered_at:
            newest[batch_id] = record
    return sorted(newest.values(), key=lambda item: item.batch_id)


def run_ptb_results(repo: Path, manifest: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["uv", "run", "awm", "ptb", "results", str(manifest), "--all", "--json"],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"ptb results failed for {manifest} (rc={completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout)


def clean_cell_keys(result: dict[str, Any]) -> set[str]:
    """Return validator-complete, unflagged cell keys for one result payload."""
    batch_id = str(result.get("batch_id", ""))
    clean: set[str] = set()
    for row in result.get("rows", []):
        attempt = row.get("completed_attempt")
        if not row.get("complete") or not isinstance(attempt, dict):
            continue
        if attempt.get("issues") or attempt.get("judge_flags"):
            continue
        clean.add(f"{batch_id}/{row['cell_id']}")
    return clean


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Retain provenance and completed rows while keeping event payloads small."""
    rows = []
    for row in result.get("rows", []):
        if not row.get("complete"):
            continue
        rows.append(
            {
                "agent": row.get("agent"),
                "agent_model": row.get("agent_model"),
                "cell_id": row.get("cell_id"),
                "complete": row.get("complete"),
                "effort": row.get("effort"),
                "replicate": row.get("replicate"),
                "completed_attempt": row.get("completed_attempt"),
            }
        )
    return {
        "batch_id": result.get("batch_id"),
        "clean_complete": result.get("clean_complete"),
        "complete": result.get("complete"),
        "flagged_complete": result.get("flagged_complete"),
        "manifest": result.get("manifest"),
        "schema_version": result.get("schema_version"),
        "spec": result.get("spec"),
        "total": result.get("total"),
        "rows": rows,
    }


def collect_results(repo: Path, records: Iterable[ManifestRecord]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for record in records:
        results[record.batch_id] = run_ptb_results(repo, record.manifest)
    return results


def _job_state(job: dict[str, Any]) -> str:
    return str(job.get("state", "")).split()[0]


def _requested_nodes(job_id: str) -> str:
    completed = subprocess.run(
        ["scontrol", "show", "job", job_id, "-o"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        return ""
    match = re.search(r"(?:^| )ReqNodeList=([^ ]+)", completed.stdout)
    return match.group(1) if match else ""


def queue_health(
    current_path: Path,
    *,
    subqueue: str = DEFAULT_SUBQUEUE,
    prefix: str = DEFAULT_PREFIX,
    expected_nodes: str = EXPECTED_NODES,
    check_routes: bool = True,
) -> dict[str, Any]:
    current = _load_json(current_path)
    sq = next((row for row in current.get("subqueues", []) if row.get("name") == subqueue), {})
    jobs: list[dict[str, Any]] = []
    for source in current.get("sources", []):
        if not str(source.get("batch_id", "")).startswith(prefix):
            continue
        source_subqueue = source.get("subqueue")
        if source_subqueue != subqueue:
            continue
        jobs.extend(source.get("jobs", []))
    running = [job for job in jobs if _job_state(job) == "RUNNING"]
    pending_by_id = {
        str(job.get("job_id")): job for job in jobs if _job_state(job) == "PENDING"
    }
    bad_routes: dict[str, str] = {}
    safe_pending = 0
    if check_routes:
        for job_id in sorted(pending_by_id):
            requested = _requested_nodes(job_id)
            if requested == expected_nodes:
                safe_pending += 1
            else:
                bad_routes[job_id] = requested or "unresolved"
    else:
        safe_pending = len(pending_by_id)
    return {
        "checked_at": utc_now(),
        "snapshot_updated_at": current.get("updated_at"),
        "subqueue": subqueue,
        "gpus_allocated": sq.get("gpus_allocated"),
        "gpus_total": sq.get("gpus_total"),
        "running": len(running),
        "pending": len(pending_by_id),
        "safe_pending": safe_pending,
        "route_bad": bad_routes,
        "expected_nodes": expected_nodes,
    }


def initial_state(clean_cells: set[str]) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "seeded_at": now,
        "updated_at": now,
        "analyzed_cells": sorted(clean_cells),
        "claims": {},
        "first_seen_new": {},
        "analysis_failures": {},
        "ready_events": [],
    }


def should_trigger(
    new_cells: set[str],
    first_seen: dict[str, float],
    *,
    now: float,
    min_new: int,
    min_partial: int,
    max_wait_seconds: int,
) -> tuple[bool, str]:
    if len(new_cells) >= min_new:
        return True, f"new_clean_complete>={min_new}"
    ages = [now - float(first_seen[cell]) for cell in new_cells if cell in first_seen]
    if len(new_cells) >= min_partial and ages and max(ages) >= max_wait_seconds:
        return True, f"partial_clean_complete>={min_partial}_aged_{max_wait_seconds}s"
    return False, ""


def analysis_prompt(payload_path: Path, snapshot_path: Path) -> str:
    return f"""ultracode

You are the read-only diagnostic partner for the online WMA evolution loop on
branch `gangda_wma_evolve`.  Analyze the newly validator-clean PTB window and
design the next evidence-efficient experiment wave.
Do not wait for or rely on GitHub PR comments.

Authoritative event payload: {payload_path}
Frozen PTB results snapshot: {snapshot_path}

First read `AGENTS.md`, `skills/wma_meta/SKILL.md`, the frozen specs named by
the snapshot, and `doc/reference/ptb_result_analysis.md`.  Then use the actual
result directories, cards, lock files, WMA verdicts, private transcripts and
`tools/wma-rca/`.  Slurm COMPLETED is not scientific completion; use only the
clean-complete cells in the payload.  Treat fewer than eight clean-complete
cells per compared arm as provisional and never recommend promotion from such
a window.
Stratify by benchmark, scientist/WMA model, public protocol and treatment mode.
Never pool raw scores across benchmarks or compare Opus 4.8 cells to the old
Opus 5 cohort as though only the WMA skill changed. Validation-only context
jobs are not scientific results. Check operator.status.json before repeating
an already executed report handoff.

Use parallel specialist analysis for every dimension that has evidence:
ledger reproduction by skill hash, uptake/timing, score levers, harm cases,
and verdict-in-lock compliance.  Reconcile their claims against exact cell,
card, manifest, receipt and result-directory citations.  Distinguish skill
causes from protocol/harness causes.

Return one self-contained report containing:
1. validity and compliance gate, including `lock.wma.state`, verdict-before-
   launch, waits/timeouts/skips, judge flags and provenance;
2. ledger by WMA skill hash and L0-L3, uptake funnel, timing, score effects,
   cost/variance, harms and PTB score comparison;
3. ranked causal diagnosis with counterevidence and uncertainty;
4. at most one proposed promotion, only if its preregistered gate is actually
   satisfied; otherwise say explicitly that none is justified;
5. the next wave as independent single-edit candidates, each with mechanism,
   primary metric, falsification, leak/cost/PTB guards, required replication,
   baseline and exact analysis gate;
6. a concise operator handoff specifying what Codex should verify, edit,
   launch and record next.

This is advisory analysis only.  Do not edit files, write commits, push, open
or comment on a PR, submit/cancel/requeue Slurm jobs, or change queue state.
"""


def claude_command(*, max_budget_usd: float) -> list[str]:
    # `ultracode` is the workflow trigger in the prompt.  Opus 5 + max is kept
    # explicit so a user-level default cannot silently downgrade the analysis.
    return [
        "claude",
        "-p",
        "--model",
        "claude-opus-5",
        "--effort",
        "max",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--permission-mode",
        "plan",
        "--permission-prompts",
        "none",
        "--max-budget-usd",
        str(max_budget_usd),
    ]


def _event_id(cells: Iterable[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(cells)).encode()).hexdigest()[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{digest}"


def _git_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    return completed.stdout.strip()


def _git_branch(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    return completed.stdout.strip()


def create_event(
    *,
    repo: Path,
    state_dir: Path,
    new_cells: set[str],
    reason: str,
    results: dict[str, dict[str, Any]],
    records: list[ManifestRecord],
) -> tuple[str, Path]:
    event_id = _event_id(new_cells)
    event_dir = state_dir / "events" / event_id
    event_dir.mkdir(parents=True, exist_ok=False)
    record_by_batch = {record.batch_id: record for record in records}
    batches = sorted({cell.split("/", 1)[0] for cell in new_cells})
    snapshot = {batch: compact_result(results[batch]) for batch in batches}
    snapshot_path = event_dir / "results.snapshot.json"
    _atomic_json(snapshot_path, snapshot)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "created_at": utc_now(),
        "branch": DEFAULT_BRANCH,
        "repo": str(repo),
        "repo_head": _git_head(repo),
        "trigger": reason,
        "clean_complete_cells": sorted(new_cells),
        "batches": [
            {
                "batch_id": batch,
                "manifest": str(record_by_batch[batch].manifest),
                "receipt": str(record_by_batch[batch].receipt),
                "spec": results[batch].get("spec"),
                "clean_complete": results[batch].get("clean_complete"),
                "complete": results[batch].get("complete"),
                "total": results[batch].get("total"),
            }
            for batch in batches
        ],
        "snapshot": str(snapshot_path),
    }
    payload_path = event_dir / "payload.json"
    _atomic_json(payload_path, payload)
    (event_dir / "prompt.txt").write_text(analysis_prompt(payload_path, snapshot_path))
    return event_id, event_dir


def launch_event_runner(
    *,
    repo: Path,
    event_dir: Path,
    state_dir: Path,
    max_budget_usd: float,
) -> int:
    stdout = (event_dir / "runner.stdout.log").open("ab")
    stderr = (event_dir / "runner.stderr.log").open("ab")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_analyze",
        "--repo",
        str(repo),
        "--state-dir",
        str(state_dir),
        "--event-dir",
        str(event_dir),
        "--max-budget-usd",
        str(max_budget_usd),
    ]
    process = subprocess.Popen(
        command,
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    stdout.close()
    stderr.close()
    _atomic_json(
        event_dir / "runner.json",
        {"pid": process.pid, "started_at": utc_now(), "command": command},
    )
    return process.pid


def analyze_event(args: argparse.Namespace) -> int:
    event_dir = args.event_dir.resolve()
    prompt = (event_dir / "prompt.txt").read_text()
    command = claude_command(max_budget_usd=args.max_budget_usd)
    _atomic_json(
        event_dir / "claude.command.json",
        {
            "command": command,
            "model": "claude-opus-5",
            "effort": "max",
            "workflow": "ultracode",
            "started_at": utc_now(),
        },
    )
    try:
        completed = subprocess.run(
            command,
            cwd=args.repo,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=args.analysis_timeout,
            check=False,
        )
        (event_dir / "claude.output.json").write_text(completed.stdout)
        (event_dir / "claude.stderr.log").write_text(completed.stderr)
        status = {
            "finished_at": utc_now(),
            "returncode": completed.returncode,
            "state": "ready" if completed.returncode == 0 else "failed",
        }
    except subprocess.TimeoutExpired as exc:
        if isinstance(exc.stdout, str):
            (event_dir / "claude.output.json").write_text(exc.stdout)
        if isinstance(exc.stderr, str):
            (event_dir / "claude.stderr.log").write_text(exc.stderr)
        status = {"finished_at": utc_now(), "returncode": 124, "state": "timed_out"}
    _atomic_json(event_dir / "status.json", status)
    return 0 if status["state"] == "ready" else 1


def _finalize_claims(state: dict[str, Any], state_dir: Path) -> list[str]:
    newly_ready: list[str] = []
    for event_id, claim in list(state.get("claims", {}).items()):
        event_dir = state_dir / "events" / event_id
        status = _load_json(event_dir / "status.json", {})
        if status.get("state") == "ready":
            state.setdefault("analyzed_cells", []).extend(claim.get("cells", []))
            state["analyzed_cells"] = sorted(set(state["analyzed_cells"]))
            state.setdefault("ready_events", []).append(event_id)
            state["ready_events"] = list(dict.fromkeys(state["ready_events"]))
            del state["claims"][event_id]
            newly_ready.append(event_id)
            continue
        if status.get("state") in {"failed", "timed_out"}:
            failures = state.setdefault("analysis_failures", {})
            failures[event_id] = status
            del state["claims"][event_id]
            continue
        runner = _load_json(event_dir / "runner.json", {})
        if runner and not _alive(runner.get("pid")):
            status = {"finished_at": utc_now(), "state": "failed", "reason": "runner_died"}
            _atomic_json(event_dir / "status.json", status)
            state.setdefault("analysis_failures", {})[event_id] = status
            del state["claims"][event_id]
    return newly_ready


def _claimed_cells(state: dict[str, Any]) -> set[str]:
    return {
        cell
        for claim in state.get("claims", {}).values()
        for cell in claim.get("cells", [])
    }


def hook_once(args: argparse.Namespace) -> dict[str, Any]:
    args.state_dir.mkdir(parents=True, exist_ok=True)
    branch = _git_branch(args.repo)
    if branch != DEFAULT_BRANCH:
        raise RuntimeError(f"hook requires branch {DEFAULT_BRANCH}, found {branch or 'detached HEAD'}")
    records = discover_manifests(args.registry, subqueue=args.subqueue, prefix=args.prefix)
    results = collect_results(args.repo, records)
    all_clean = set().union(*(clean_cell_keys(result) for result in results.values()))
    state_path = args.state_dir / "state.json"
    if not state_path.exists():
        state = initial_state(all_clean)
        _atomic_json(state_path, state)
        seeded = True
    else:
        state = _load_json(state_path)
        seeded = False

    newly_ready = _finalize_claims(state, args.state_dir)
    known = set(state.get("analyzed_cells", [])) | _claimed_cells(state)
    new_cells = all_clean - known
    now = time.time()
    first_seen = state.setdefault("first_seen_new", {})
    for cell in new_cells:
        first_seen.setdefault(cell, now)
    for cell in list(first_seen):
        if cell not in new_cells:
            del first_seen[cell]

    health = queue_health(
        args.current,
        subqueue=args.subqueue,
        prefix=args.prefix,
        expected_nodes=args.expected_nodes,
        check_routes=not args.skip_route_check,
    )
    _atomic_json(args.state_dir / "queue-health.json", health)

    launched: dict[str, Any] | None = None
    trigger, reason = should_trigger(
        new_cells,
        first_seen,
        now=now,
        min_new=args.min_new,
        min_partial=args.min_partial,
        max_wait_seconds=args.max_wait_seconds,
    )
    if trigger and not state.get("claims"):
        event_id, event_dir = create_event(
            repo=args.repo,
            state_dir=args.state_dir,
            new_cells=new_cells,
            reason=reason,
            results=results,
            records=records,
        )
        if args.no_claude:
            pid = None
            _atomic_json(
                event_dir / "status.json",
                {"state": "dry_run", "finished_at": utc_now()},
            )
        else:
            state.setdefault("claims", {})[event_id] = {
                "cells": sorted(new_cells),
                "claimed_at": utc_now(),
                "reason": reason,
            }
            pid = launch_event_runner(
                repo=args.repo,
                event_dir=event_dir,
                state_dir=args.state_dir,
                max_budget_usd=args.max_budget_usd,
            )
        launched = {"event_id": event_id, "event_dir": str(event_dir), "pid": pid}

    state["updated_at"] = utc_now()
    state["last_check"] = {
        "all_clean_complete": len(all_clean),
        "new_clean_complete": len(new_cells),
        "manifest_count": len(records),
        "queue": health,
    }
    _atomic_json(state_path, state)
    summary = {
        "checked_at": state["updated_at"],
        "seeded": seeded,
        "manifest_count": len(records),
        "all_clean_complete": len(all_clean),
        "new_clean_complete": len(new_cells),
        "newly_ready": newly_ready,
        "launched": launched,
        "queue": health,
    }
    _atomic_json(args.state_dir / "status.json", summary)
    return summary


def _health_alert(summary: dict[str, Any], replenish_threshold: int) -> list[str]:
    health = summary["queue"]
    alerts = []
    if health.get("gpus_allocated") != 16:
        alerts.append(f"owned_gpus={health.get('gpus_allocated')}")
    if health.get("safe_pending", 0) < replenish_threshold:
        alerts.append(f"safe_pending={health.get('safe_pending')}")
    if health.get("safe_pending", 0) <= 8:
        alerts.append("HARD_PENDING_FLOOR")
    if health.get("route_bad"):
        alerts.append(f"route_bad={len(health['route_bad'])}")
    return alerts


def run_daemon(args: argparse.Namespace) -> int:
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_dir / "daemon.lock"
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"daemon already running: {lock_path}", file=sys.stderr)
        return 2
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    _atomic_json(
        args.state_dir / "daemon.json",
        {"pid": os.getpid(), "started_at": utc_now(), "interval_seconds": args.interval},
    )
    last_alerts: list[str] = []
    while not stopping:
        try:
            summary = hook_once(args)
            alerts = _health_alert(summary, args.replenish_threshold)
            if alerts != last_alerts:
                label = "HEALTH_EVENT" if alerts else "HEALTH_RECOVERED"
                print(f"{utc_now()} {label} {' '.join(alerts)}", flush=True)
                last_alerts = alerts
            if summary["launched"]:
                print(f"{utc_now()} ANALYSIS_LAUNCHED {json.dumps(summary['launched'])}", flush=True)
            for event_id in summary["newly_ready"]:
                print(f"{utc_now()} ANALYSIS_READY {event_id}", flush=True)
        except Exception as exc:  # noqa: BLE001 - daemon survives transient external faults
            print(f"{utc_now()} MONITOR_ERROR {type(exc).__name__}: {exc}", flush=True)
        deadline = time.monotonic() + args.interval
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    _atomic_json(args.state_dir / "daemon.json", {"pid": os.getpid(), "stopped_at": utc_now()})
    return 0


def show_status(args: argparse.Namespace) -> int:
    daemon = _load_json(args.state_dir / "daemon.json", {})
    status = _load_json(args.state_dir / "status.json", {})
    state = _load_json(args.state_dir / "state.json", {})
    print(
        json.dumps(
            {
                "daemon": {**daemon, "alive": _alive(daemon.get("pid"))},
                "status": status,
                "active_claims": state.get("claims", {}),
                "ready_events": state.get("ready_events", []),
                "analysis_failures": state.get("analysis_failures", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--subqueue", default=DEFAULT_SUBQUEUE)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--expected-nodes", default=EXPECTED_NODES)
    parser.add_argument("--min-new", type=int, default=8)
    parser.add_argument("--min-partial", type=int, default=4)
    parser.add_argument("--max-wait-seconds", type=int, default=21600)
    parser.add_argument("--max-budget-usd", type=float, default=25.0)
    parser.add_argument("--skip-route-check", action="store_true")
    parser.add_argument("--no-claude", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    once = commands.add_parser("once", help="check once and possibly launch one analysis")
    _common(once)
    run = commands.add_parser("run", help="run the slow singleton daemon")
    _common(run)
    run.add_argument("--interval", type=int, default=3600)
    run.add_argument("--replenish-threshold", type=int, default=24)
    status = commands.add_parser("status", help="show daemon, queue and analysis status")
    status.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    analyze = commands.add_parser("_analyze", help=argparse.SUPPRESS)
    analyze.add_argument("--repo", type=Path, required=True)
    analyze.add_argument("--state-dir", type=Path, required=True)
    analyze.add_argument("--event-dir", type=Path, required=True)
    analyze.add_argument("--max-budget-usd", type=float, required=True)
    analyze.add_argument("--analysis-timeout", type=int, default=14400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "once":
        print(json.dumps(hook_once(args), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        return run_daemon(args)
    if args.command == "status":
        return show_status(args)
    if args.command == "_analyze":
        return analyze_event(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
